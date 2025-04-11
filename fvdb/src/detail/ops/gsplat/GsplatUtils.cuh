// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GSPLATUTILS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GSPLATUTILS_CUH

#include "CameraProjections.cuh"
#include "GsplatTypes.cuh"

#include <nanovdb/math/Math.h>

namespace fvdb {
namespace detail {
namespace ops {

template <class T>
inline __device__ uint32_t
bin_search(const T *arr, const uint32_t len, const T val) {
    uint32_t low = 0, high = len - 1;
    while (low <= high) {
        const uint32_t mid = (low + high) / 2;
        if (arr[mid] <= val) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return low - 1;
}

template <typename T>
inline __device__ mat3<T>
                  quat_to_rotmat(const vec4<T> quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    T inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    T x2 = x * x, y2 = y * y, z2 = z * z;
    T xy = x * y, xz = x * z, yz = y * z;
    T wx = w * x, wy = w * y, wz = w * z;
    return mat3<T>((1.f - 2.f * (y2 + z2)), (2.f * (xy + wz)),
                                     (2.f * (xz - wy)),      // 1st col
                                     (2.f * (xy - wz)), (1.f - 2.f * (x2 + z2)),
                                     (2.f * (yz + wx)),      // 2nd col
                                     (2.f * (xz + wy)), (2.f * (yz - wx)),
                                     (1.f - 2.f * (x2 + y2)) // 3rd col
                      );
}

/// @brief Convert a quaternion to a rotation matrix
/// @param quat: the quaternion
/// @return the rotation matrix
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
                  quatToRotMat(nanovdb::math::Vec4<T> const &quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    T inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    T x2 = x * x, y2 = y * y, z2 = z * z;
    T xy = x * y, xz = x * z, yz = y * z;
    T wx = w * x, wy = w * y, wz = w * z;
    return nanovdb::math::Mat3<T>(
        (1.f - 2.f * (y2 + z2)), (2.f * (xy - wz)), (2.f * (xz + wy)), // 1st row
        (2.f * (xy + wz)), (1.f - 2.f * (x2 + z2)), (2.f * (yz - wx)), // 2nd row
        (2.f * (xz - wy)), (2.f * (yz + wx)), (1.f - 2.f * (x2 + y2))  // 3rd row
    );
}

template <typename T>
inline __device__ void
quat_to_rotmat_vjp(const vec4<T> quat, const mat3<T> v_R, vec4<T> &v_quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    T inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    vec4<T> v_quat_n =
        vec4<T>(2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                       z * (v_R[0][1] - v_R[1][0])),
                2.f * (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
                       z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
                2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
                       z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
                2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
                       2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])));

    vec4<T> quat_n = vec4<T>(w, x, y, z);
    v_quat += (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

template <typename T>
inline __device__ void
quat_scale_to_covar_vjp(
    // fwd inputs
    const vec4<T> quat, const vec3<T> scale,
    // precompute
    const mat3<T> R,
    // grad outputs
    const mat3<T> v_covar,
    // grad inputs
    vec4<T> &v_quat, vec3<T> &v_scale) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = scale[0], sy = scale[1], sz = scale[2];

    // M = R * S
    mat3<T> S = mat3<T>(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3<T> M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3<T> v_M = (v_covar + glm::transpose(v_covar)) * M;
    mat3<T> v_R = v_M * S;

    // grad for (quat, scale) from covar
    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);

    v_scale[0] += R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2];
    v_scale[1] += R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2];
    v_scale[2] += R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2];
}

template <typename T>
inline __device__ void
quat_scale_to_covar_preci(const vec4<T> quat, const vec3<T> scale,
                          // optional outputs
                          mat3<T> *covar, mat3<T> *preci) {
    mat3<T> R = quat_to_rotmat<T>(quat);
    if (covar != nullptr) {
        // C = R * S * S * Rt
        mat3<T> S = mat3<T>(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        mat3<T> M = R * S;
        *covar    = M * glm::transpose(M);
    }
    if (preci != nullptr) {
        // P = R * S^-1 * S^-1 * Rt
        mat3<T> S = mat3<T>(1.0f / scale[0], 0.f, 0.f, 0.f, 1.0f / scale[1], 0.f, 0.f, 0.f,
                            1.0f / scale[2]);
        mat3<T> M = R * S;
        *preci    = M * glm::transpose(M);
    }
}

/// @brief Convert a quaternion and scale to a covariance and precision matrix
/// @param quat: the quaternion
/// @param scale: the scale
/// @param outCovar: the covariance matrix
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
quatAndScaleToCovariance(const nanovdb::math::Vec4<T> &quat, const nanovdb::math::Vec3<T> &scale) {
    nanovdb::math::Mat3<T> const R = quatToRotMat<T>(quat);
    // C = R * S * S * Rt
    nanovdb::math::Mat3<T> S(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
    nanovdb::math::Mat3<T> M = R * S;
    return M * M.transpose();
}

template <typename T>
inline __device__ T
add_blur(const T eps2d, mat2<T> &covar, T &compensation) {
    T det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    T det_blur   = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

template <typename T>
inline __device__ T
add_blur(const T eps2d, nanovdb::math::Mat2<T> &outCovar, T &outCompensation) {
    const T det_orig = outCovar[0][0] * outCovar[1][1] - outCovar[0][1] * outCovar[1][0];
    outCovar[0][0] += eps2d;
    outCovar[1][1] += eps2d;
    const T det_blur = outCovar[0][0] * outCovar[1][1] - outCovar[0][1] * outCovar[1][0];
    outCompensation  = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

template <typename T>
inline __device__ void
add_blur_vjp(const T eps2d, const mat2<T> conic_blur, const T compensation, const T v_compensation,
             mat2<T> &v_covar) {
    // comp = sqrt(det(covar) / det(covar_blur))

    // d [det(M)] / d M = adj(M)
    // d [det(M + aI)] / d M  = adj(M + aI) = adj(M) + a * I
    // d [det(M) / det(M + aI)] / d M
    // = (det(M + aI) * adj(M) - det(M) * adj(M + aI)) / (det(M + aI))^2
    // = adj(M) / det(M + aI) - adj(M + aI) / det(M + aI) * comp^2
    // = (adj(M) - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M + aI) = adj(M) + a * I
    // = (adj(M + aI) - aI - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M) / det(M) = inv(M)
    // = (1 - comp^2) * inv(M + aI) - aI / det(M + aI)
    // given det(inv(M)) = 1 / det(M)
    // = (1 - comp^2) * inv(M + aI) - aI * det(inv(M + aI))
    // = (1 - comp^2) * conic_blur - aI * det(conic_blur)

    T det_conic_blur = conic_blur[0][0] * conic_blur[1][1] - conic_blur[0][1] * conic_blur[1][0];
    T v_sqr_comp     = v_compensation * 0.5 / (compensation + 1e-6);
    T one_minus_sqr_comp = 1 - compensation * compensation;
    v_covar[0][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] - eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] - eps2d * det_conic_blur);
}

template <typename T>
inline __device__ void
pos_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3<T> R, const vec3<T> t, const vec3<T> p, vec3<T> &p_c) {
    p_c = R * p + t;
}

// @brief Transform a point from world to camera coordinates
// @param camToWorldRotation: the rotation matrix from camera to world
// @param camToWorldTranslation: the translation vector from camera to world
// @param worldSpacePoint: the point in world coordinates
template <typename T>
inline __device__ nanovdb::math::Vec3<T>
                  transformPointWorldToCam(nanovdb::math::Mat3<T> const &camToWorldRotation,
                                           nanovdb::math::Vec3<T> const &camToWorldTranslation,
                                           nanovdb::math::Vec3<T> const &worldSpacePoint) {
    return camToWorldRotation * worldSpacePoint + camToWorldTranslation;
}

template <typename T>
inline __device__ void
pos_world_to_cam_vjp(
    // fwd inputs
    const mat3<T> R, const vec3<T> t, const vec3<T> p,
    // grad outputs
    const vec3<T> v_p_c,
    // grad inputs
    mat3<T> &v_R, vec3<T> &v_t, vec3<T> &v_p) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    v_R += glm::outerProduct(v_p_c, p);
    v_t += v_p_c;
    v_p += glm::transpose(R) * v_p_c;
}

template <typename T>
inline __device__ void
covar_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3<T> R, const mat3<T> covar, mat3<T> &covar_c) {
    covar_c = R * covar * glm::transpose(R);
}

/// @brief Transform a covariance matrix from world to camera coordinates
/// @param R: the rotation matrix from world to camera
/// @param covar: the covariance matrix in world coordinates
/// @return the covariance matrix in camera coordinates
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
                  transformCovarianceWorldToCam(nanovdb::math::Mat3<T> const &R,
                                                nanovdb::math::Mat3<T> const &covar) {
    return R * covar * R.transpose();
}

template <typename T>
inline __device__ void
covar_world_to_cam_vjp(
    // fwd inputs
    const mat3<T> R, const mat3<T> covar,
    // grad outputs
    const mat3<T> v_covar_c,
    // grad inputs
    mat3<T> &v_R, mat3<T> &v_covar) {
    // for D = W * X * WT, G = df/dD
    // df/dX = WT * G * W
    // df/dW
    // = G * (X * WT)T + ((W * X)T * G)T
    // = G * W * XT + (XT * WT * G)T
    // = G * W * XT + GT * W * X
    v_R += v_covar_c * R * glm::transpose(covar) + glm::transpose(v_covar_c) * R * covar;
    v_covar += glm::transpose(R) * v_covar_c * R;
}

template <typename T>
inline __device__ void
inverse_vjp(const T Minv, const T v_Minv, T &v_M) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    v_M += -Minv * v_Minv * Minv;
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GSPLATUTILS_CUH
