// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAPROJECTIONS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAPROJECTIONS_CUH

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
inline __device__ std::tuple<nanovdb::math::Mat2<T>, nanovdb::math::Vec2<T>>
projectGaussianPerspective(const nanovdb::math::Vec3<T> &mean3d,
                           const nanovdb::math::Mat3<T> &cov3d,
                           const T fx,
                           const T fy,
                           const T cx,
                           const T cy,
                           const int64_t width,
                           const int64_t height) {
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

    const Mat2x3 J(fx * rz,
                   0.f,
                   -fx * tx * rz2, // 1st row
                   0.f,
                   fy * rz,
                   -fy * ty * rz2  // 2nd row
    );

    const Mat2 cov2d = J * cov3d * J.transpose();
    const Vec2 mean2d({fx * x * rz + cx, fy * y * rz + cy});

    return {cov2d, mean2d};
}

// Compute the Jacobian-vector product of the perspective projection operator which maps 3D
// gaussians to 2D pixel space gaussians. The jacobian is with respect to the 3D mean and covariance
// matrix. The projection is defined by the camera intrinsics fx, fy, cx, cy and the image
// dimensions width and height.
template <typename T>
inline __device__ std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Vec3<T>>
projectGaussianPerspectiveVectorJacobianProduct(const nanovdb::math::Vec3<T> &mean3d,
                                                const nanovdb::math::Mat3<T> &cov3d,
                                                const T fx,
                                                const T fy,
                                                const T cx,
                                                const T cy,
                                                const int64_t width,
                                                const int64_t height,
                                                const nanovdb::math::Mat2<T> &dLossDCovar2d,
                                                const nanovdb::math::Vec2<T> &dLossDMean2d) {
    using Mat2x3 = nanovdb::math::Mat2x3<T>;
    using Mat3   = nanovdb::math::Mat3<T>;
    using Vec3   = nanovdb::math::Vec3<T>;

    const T x = mean3d[0];
    const T y = mean3d[1];
    const T z = mean3d[2];

    const T tanFovX = 0.5f * width / fx;
    const T tanFovY = 0.5f * height / fy;
    const T limXPos = (width - cx) / fx + 0.3f * tanFovX;
    const T limXNeg = cx / fx + 0.3f * tanFovX;
    const T limYPos = (height - cy) / fy + 0.3f * tanFovY;
    const T limYNeg = cy / fy + 0.3f * tanFovY;

    const T rz  = 1.f / z;
    const T rz2 = rz * rz;
    const T tx  = z * min(limXPos, max(-limXNeg, x * rz));
    const T ty  = z * min(limYPos, max(-limYNeg, y * rz));

    const Mat2x3 J(fx * rz,
                   0.f,
                   -fx * tx * rz2, // 1st row
                   0.f,
                   fy * rz,
                   -fy * ty * rz2  // 2nd row
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    const Mat3 dLossDCovar3d(J.transpose() * dLossDCovar2d * J);

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    Vec3 dLossDMean3d(fx * rz * dLossDMean2d[0],
                      fy * rz * dLossDMean2d[1],
                      -(fx * x * dLossDMean2d[0] + fy * y * dLossDMean2d[1]) * rz2);

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    const T rz3 = rz2 * rz;
    const Mat2x3 dLossDJ =
        dLossDCovar2d * J * cov3d.transpose() + dLossDCovar2d.transpose() * J * cov3d;

    // fov clipping
    if (x * rz <= limXPos && x * rz >= -limXNeg) {
        dLossDMean3d[0] += -fx * rz2 * dLossDJ[0][2];
    } else {
        dLossDMean3d[2] += -fx * rz3 * dLossDJ[0][2] * tx;
    }
    if (y * rz <= limYPos && y * rz >= -limYNeg) {
        dLossDMean3d[1] += -fy * rz2 * dLossDJ[1][2];
    } else {
        dLossDMean3d[2] += -fy * rz3 * dLossDJ[1][2] * ty;
    }
    dLossDMean3d[2] += -fx * rz2 * dLossDJ[0][0] - fy * rz2 * dLossDJ[1][1] +
                       2.f * fx * tx * rz3 * dLossDJ[0][2] + 2.f * fy * ty * rz3 * dLossDJ[1][2];

    return {dLossDCovar3d, dLossDMean3d};
}

// Apply orthographic projection to a 3D Gaussian defined by it's 3D mean and 3x3 covariance matrix.
// The projection is defined by the camera intrinsics fx, fy, cx, cy and the image dimensions width
// and height. The function returns the 2D mean and 2x2 covariance matrix of the projected Gaussian
// in the pixel coordinates of the image.
template <typename T>
inline __device__ std::tuple<nanovdb::math::Mat2<T>, nanovdb::math::Vec2<T>>
projectGaussianOrthographic(const nanovdb::math::Vec3<T> &mean3d,
                            const nanovdb::math::Mat3<T> &cov3d,
                            const T fx,
                            const T fy,
                            const T cx,
                            const T cy,
                            const uint32_t width,
                            const uint32_t height) {
    using Mat2x3 = nanovdb::math::Mat2x3<T>;
    using Mat2   = nanovdb::math::Mat2<T>;
    using Vec2   = nanovdb::math::Vec2<T>;

    const T x = mean3d[0];
    const T y = mean3d[1];
    const T z = mean3d[2];

    const Mat2x3 J(fx,
                   0.f,
                   0.f, // 1st row
                   0.f,
                   fy,
                   0.f  // 2nd row
    );

    const Mat2 cov2d = J * cov3d * J.transpose();
    const Vec2 mean2d({fx * x + cx, fy * y + cy});

    return {cov2d, mean2d};
}

// Compute the Jacobian-vector product of the orthographic projection operator which maps 3D
// gaussians to 2D pixel space gaussians. The jacobian is with respect to the 3D mean and covariance
// matrix. The projection is defined by the camera intrinsics fx, fy, cx, cy and the image
// dimensions width and height.
template <typename T>
inline __device__ std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Vec3<T>>
projectGaussianOrthographicVectorJacobianProduct(const nanovdb::math::Vec3<T> &mean3d,
                                                 const nanovdb::math::Mat3<T> &cov3d,
                                                 const T fx,
                                                 const T fy,
                                                 const T cx,
                                                 const T cy,
                                                 const int64_t width,
                                                 const int64_t height,
                                                 const nanovdb::math::Mat2<T> &dLossDCovar2d,
                                                 const nanovdb::math::Vec2<T> &dLossDMean2d) {
    using Mat2x3 = nanovdb::math::Mat2x3<T>;
    using Mat3   = nanovdb::math::Mat3<T>;
    using Vec3   = nanovdb::math::Vec3<T>;

    const T x = mean3d[0];
    const T y = mean3d[1];
    const T z = mean3d[2];

    const Mat2x3 J(fx,
                   0.f,
                   0.f, // 1st row
                   0.f,
                   fy,
                   0.f  // 2nd row
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    const Mat3 dLossDCovar3d(J.transpose() * dLossDCovar2d * J);

    // df/dx = fx * df/dpixx
    // df/dy = fy * df/dpixy
    // df/dz = 0
    const Vec3 dLossDMean3d(fx * dLossDMean2d[0], fy * dLossDMean2d[1], 0.f);

    return {dLossDCovar3d, dLossDMean3d};
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAPROJECTIONS_CUH
