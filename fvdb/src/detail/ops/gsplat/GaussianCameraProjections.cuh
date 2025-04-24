// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAPROJECTIONS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAPROJECTIONS_CUH

#include "GaussianTypes.cuh"

#include <nanovdb/math/Math.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

// Apply perspective projection to a 3D Gaussian defined by it's 3D mean and 3x3 covariance matrix.
// The projection is defined by the camera intrinsics fx, fy, cx, cy and the image dimensions width
// and height. The function returns the 2D mean and 2x2 covariance matrix of the projected Gaussian
// in the pixel coordinates of the image.
template <typename T>
inline __device__ std::tuple<mat2<T>, vec2<T>>
persp_proj(const vec3<T> mean3d, const mat3<T> cov3d, const T fx, const T fy, const T cx,
           const T cy, const uint32_t width, const uint32_t height) {
    const T x = mean3d[0];
    const T y = mean3d[1];
    const T z = mean3d[2];

    const T tan_fovx  = 0.5f * width / fx;
    const T tan_fovy  = 0.5f * height / fy;
    const T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    const T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    const T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    const T rz  = 1.f / z;
    const T rz2 = rz * rz;
    const T tx  = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    const T ty  = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    const auto J = mat3x2<T>{
        fx * rz,
        0.f,           // 1st column
        0.f,
        fy * rz,       // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    };

    const mat2<T> cov2d  = J * cov3d * glm::transpose(J);
    const vec2<T> mean2d = vec2<T>({ fx * x * rz + cx, fy * y * rz + cy });

    return { cov2d, mean2d };
}

// Apply perspective projection to a 3D Gaussian defined by it's 3D mean and 3x3 covariance matrix.
// The projection is defined by the camera intrinsics fx, fy, cx, cy and the image dimensions width
// and height. The function returns the 2D mean and 2x2 covariance matrix of the projected Gaussian
// in the pixel coordinates of the image.
template <typename T>
inline __device__ std::tuple<nanovdb::math::Mat2<T>, nanovdb::math::Vec2<T>>
                  projectGaussianPerspective(const nanovdb::math::Vec3<T> &mean3d,
                                             const nanovdb::math::Mat3<T> &cov3d, const T fx, const T fy, const T cx,
                                             const T cy, const int64_t width, const int64_t height) {
    using Mat2x3 = nanovdb::math::Mat2x3<T>;
    using Mat2   = nanovdb::math::Mat2<T>;
    using Vec2   = nanovdb::math::Vec2<T>;

    const T x = mean3d[0];
    const T y = mean3d[1];
    const T z = mean3d[2];

    const T tanFovX = T(0.5) * width / fx;
    const T tanFovY = T(0.5) * height / fy;
    const T limXPos = (width - cx) / fx + T(0.3) * tanFovX;
    const T limXNeg = cx / fx + T(0.3) * tanFovX;
    const T limYPos = (height - cy) / fy + T(0.3) * tanFovY;
    const T limYNeg = cy / fy + T(0.3) * tanFovY;

    const T rz  = T(1.0) / z;
    const T rz2 = rz * rz;
    const T tx  = z * min(limXPos, max(-limXNeg, x * rz));
    const T ty  = z * min(limYPos, max(-limYNeg, y * rz));

    const Mat2x3 J(fx * rz, 0.f, -fx * tx * rz2, // 1st row (was 1st column)
                                     0.f, fy * rz, -fy * ty * rz2  // 2nd row (was 2nd column)
                      );

    const Mat2 cov2d = J * cov3d * J.transpose();
    const Vec2 mean2d({ fx * x * rz + cx, fy * y * rz + cy });

    return { cov2d, mean2d };
}

// Compute the Jacobian-vector product of the perspective projection operator which maps 3D
// gaussians to 2D pixel space gaussians. The jacobian is with respect to the 3D mean and covariance
// matrix. The projection is defined by the camera intrinsics fx, fy, cx, cy and the image
// dimensions width and height.
template <typename T>
inline __device__ std::tuple<mat3<T>, vec3<T>>
persp_proj_vjp(const vec3<T> mean3d, const mat3<T> cov3d, const T fx, const T fy, const T cx,
               const T cy, const uint32_t width, const uint32_t height, const mat2<T> v_cov2d,
               const vec2<T> v_mean2d) {
    const T x = mean3d[0];
    const T y = mean3d[1];
    const T z = mean3d[2];

    const T tan_fovx  = 0.5f * width / fx;
    const T tan_fovy  = 0.5f * height / fy;
    const T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    const T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    const T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    const T rz  = 1.f / z;
    const T rz2 = rz * rz;
    const T tx  = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    const T ty  = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    const auto J = mat3x2<T>{
        fx * rz,
        0.f,           // 1st column
        0.f,
        fy * rz,       // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    };

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    mat3<T> v_cov3d = glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    auto v_mean3d = vec3<T>(fx * rz * v_mean2d[0], fy * rz * v_mean2d[1],
                            -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2);

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    const T         rz3 = rz2 * rz;
    const mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) + glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] + 2.f * fx * tx * rz3 * v_J[2][0] +
                  2.f * fy * ty * rz3 * v_J[2][1];

    return { v_cov3d, v_mean3d };
}

// Apply orthographic projection to a 3D Gaussian defined by it's 3D mean and 3x3 covariance matrix.
// The projection is defined by the camera intrinsics fx, fy, cx, cy and the image dimensions width
// and height. The function returns the 2D mean and 2x2 covariance matrix of the projected Gaussian
// in the pixel coordinates of the image.
template <typename T>
inline __device__ std::tuple<mat2<T>, vec2<T>>
ortho_proj(const vec3<T> mean3d, const mat3<T> cov3d, const T fx, const T fy, const T cx,
           const T cy, const uint32_t width, const uint32_t height) {
    const T x = mean3d[0];
    const T y = mean3d[1];
    const T z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    const auto J = mat3x2<T>{
        fx,
        0.f, // 1st column
        0.f,
        fy,  // 2nd column
        0.f,
        0.f  // 3rd column
    };

    const mat2<T> cov2d  = J * cov3d * glm::transpose(J);
    const vec2<T> mean2d = vec2<T>({ fx * x + cx, fy * y + cy });

    return { cov2d, mean2d };
}

// Apply orthographic projection to a 3D Gaussian defined by it's 3D mean and 3x3 covariance matrix.
// The projection is defined by the camera intrinsics fx, fy, cx, cy and the image dimensions width
// and height. The function returns the 2D mean and 2x2 covariance matrix of the projected Gaussian
// in the pixel coordinates of the image.
template <typename T>
inline __device__ std::tuple<nanovdb::math::Mat2<T>, nanovdb::math::Vec2<T>>
                  projectGaussianOrthographic(const nanovdb::math::Vec3<T> &mean3d,
                                              const nanovdb::math::Mat3<T> &cov3d, const T fx, const T fy, const T cx,
                                              const T cy, const uint32_t width, const uint32_t height) {
    using Mat2x3 = nanovdb::math::Mat2x3<T>;
    using Mat2   = nanovdb::math::Mat2<T>;
    using Vec2   = nanovdb::math::Vec2<T>;

    const T x = mean3d[0];
    const T y = mean3d[1];
    const T z = mean3d[2];

    const Mat2x3 J(fx, 0.f, 0.f, // 1st row
                                     0.f, fy, 0.f  // 2nd row
                      );

    const Mat2 cov2d = J * cov3d * J.transpose();
    const Vec2 mean2d({ fx * x + cx, fy * y + cy });

    return { cov2d, mean2d };
}

// Compute the Jacobian-vector product of the orthographic projection operator which maps 3D
// gaussians to 2D pixel space gaussians. The jacobian is with respect to the 3D mean and covariance
// matrix. The projection is defined by the camera intrinsics fx, fy, cx, cy and the image
// dimensions width and height.
template <typename T>
inline __device__ std::tuple<mat3<T>, vec3<T>>
ortho_proj_vjp(const vec3<T> mean3d, const mat3<T> cov3d, const T fx, const T fy, const T cx,
               const T cy, const uint32_t width, const uint32_t height, const mat2<T> v_cov2d,
               const vec2<T> v_mean2d) {
    const T x = mean3d[0];
    const T y = mean3d[1];
    const T z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    const auto J = mat3x2<T>{
        fx,
        0.f, // 1st column
        0.f,
        fy,  // 2nd column
        0.f,
        0.f  // 3rd column
    };

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    mat3<T> v_cov3d = glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * df/dpixx
    // df/dy = fy * df/dpixy
    // df/dz = 0
    auto v_mean3d = vec3<T>(fx * v_mean2d[0], fy * v_mean2d[1], 0.f);

    return { v_cov3d, v_mean3d };
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAPROJECTIONS_CUH
