// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANUTILS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANUTILS_CUH

#include "GaussianCameraProjections.cuh"

#include <nanovdb/math/Math.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Performs a binary search on a sorted array to find the insertion point for a value
///
/// This function implements a standard binary search algorithm to find the last index
/// in the array where the element is less than or equal to the specified value.
///
/// The function assumes that the array is sorted in non-decreasing order:
/// arr[0] <= arr[1] <= ... <= arr[len-1]
///
/// Time complexity: O(log n) where n is the length of the array
///
/// @tparam T Type of elements in the array (must support comparison operators)
/// @param arr Pointer to the sorted array
/// @param len Length of the array
/// @param val Value to search for
/// @return Index of the last element that is less than or equal to val, or -1 if no such element
/// exists
template <class T>
inline __device__ uint32_t
binSearch(const T *arr, const uint32_t len, const T val) {
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

/// @brief Converts a quaternion to a 3x3 rotation matrix
///
/// This function takes a quaternion [w,x,y,z] and converts it to the equivalent
/// 3x3 rotation matrix representation. The quaternion is first normalized to ensure
/// it has unit length, which is required for a proper rotation.
///
/// The conversion uses the standard formula:
/// R = [
///   1-2(y²+z²)    2(xy-wz)      2(xz+wy)
///   2(xy+wz)      1-2(x²+z²)    2(yz-wx)
///   2(xz-wy)      2(yz+wx)      1-2(x²+y²)
/// ]
///
/// Where w,x,y,z are the components of the normalized quaternion.
///
/// @param quat Input quaternion in [w,x,y,z] format
/// @return nanovdb::math::Mat3<T> 3x3 rotation matrix equivalent to the quaternion
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
                  quaternionToRotationMatrix(nanovdb::math::Vec4<T> const &quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    T inverseNormalization = rsqrt(x * x + y * y + z * z + w * w);
    x *= inverseNormalization;
    y *= inverseNormalization;
    z *= inverseNormalization;
    w *= inverseNormalization;
    T x2 = x * x, y2 = y * y, z2 = z * z;
    T xy = x * y, xz = x * z, yz = y * z;
    T wx = w * x, wy = w * y, wz = w * z;
    return nanovdb::math::Mat3<T>(
        (1.f - 2.f * (y2 + z2)), (2.f * (xy - wz)), (2.f * (xz + wy)), // 1st row
        (2.f * (xy + wz)), (1.f - 2.f * (x2 + z2)), (2.f * (yz - wx)), // 2nd row
        (2.f * (xz - wy)), (2.f * (yz + wx)), (1.f - 2.f * (x2 + y2))  // 3rd row
    );
}

/// @brief Computes the vector-Jacobian product for quaternion to rotation matrix transformation
///
/// This function computes the gradient of the loss with respect to a quaternion (dL/dq)
/// given the gradient of the loss with respect to a rotation matrix (dL/dR) that was
/// derived from the quaternion. This is essentially a backwards pass through the quaternion
/// to rotation matrix transformation.
///
/// The function first normalizes the quaternion, computes the vector-Jacobian product
/// for the normalized quaternion, and then applies the chain rule to get the gradient
/// with respect to the original quaternion.
///
/// Mathematical details:
/// 1. Normalize quaternion to unit length
/// 2. Compute vector-Jacobian product for rotation matrix derivatives
/// 3. Project gradient to ensure it's orthogonal to the quaternion (preserving unit length)
/// 4. Apply chain rule for normalization
///
/// @param quat Input quaternion [w,x,y,z]
/// @param dLossDRotation Gradient of loss with respect to rotation matrix (dL/dR)
/// @return nanovdb::math::Vec4<T> Gradient of loss with respect to quaternion (dL/dq)
template <typename T>
inline __device__ nanovdb::math::Vec4<T>
quaternionToRotationMatrixVectorJacobianProduct(const nanovdb::math::Vec4<T> &quat,
                                                const nanovdb::math::Mat3<T> &dLossDRotation) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    const T inverseNormalization = rsqrt(x * x + y * y + z * z + w * w);
    x *= inverseNormalization;
    y *= inverseNormalization;
    z *= inverseNormalization;
    w *= inverseNormalization;
    const nanovdb::math::Vec4<T> dLossDQuatNormalized(
        2.f * (x * (dLossDRotation[2][1] - dLossDRotation[1][2]) +
               y * (dLossDRotation[0][2] - dLossDRotation[2][0]) +
               z * (dLossDRotation[1][0] - dLossDRotation[0][1])),
        2.f * (-2.f * x * (dLossDRotation[1][1] + dLossDRotation[2][2]) +
               y * (dLossDRotation[1][0] + dLossDRotation[0][1]) +
               z * (dLossDRotation[2][0] + dLossDRotation[0][2]) +
               w * (dLossDRotation[2][1] - dLossDRotation[1][2])),
        2.f * (x * (dLossDRotation[1][0] + dLossDRotation[0][1]) -
               2.f * y * (dLossDRotation[0][0] + dLossDRotation[2][2]) +
               z * (dLossDRotation[2][1] + dLossDRotation[1][2]) +
               w * (dLossDRotation[0][2] - dLossDRotation[2][0])),
        2.f * (x * (dLossDRotation[2][0] + dLossDRotation[0][2]) +
               y * (dLossDRotation[2][1] + dLossDRotation[1][2]) -
               2.f * z * (dLossDRotation[0][0] + dLossDRotation[1][1]) +
               w * (dLossDRotation[1][0] - dLossDRotation[0][1])));

    const nanovdb::math::Vec4<T> quatNormalized(w, x, y, z);
    return (dLossDQuatNormalized - dLossDQuatNormalized.dot(quatNormalized) * quatNormalized) *
           inverseNormalization;
}

/// @brief Computes gradients of loss with respect to quaternion and scale parameters
///
/// This function calculates the vector-Jacobian product for quaternion and scale parameters
/// that were used to generate a covariance matrix. It's used in the backward pass of
/// automatic differentiation when computing gradients through the covariance matrix computation.
///
/// The covariance matrix is computed as C = M * M^T where M = R * S, with:
/// - R being the rotation matrix derived from the quaternion
/// - S being the diagonal scale matrix
///
/// The function implements the chain rule to propagate gradients from the covariance matrix
/// back to the quaternion and scale parameters.
///
/// Mathematical details:
/// 1. For matrix operations D = M * M^T, the gradient follows:
///    dL/dM = (dL/dD + (dL/dD)^T) * M
/// 2. For D = R * S, the gradient follows:
///    dL/dR = (dL/dD) * S^T and dL/dS = R^T * (dL/dD)
///
/// @param quat Input quaternion [w,x,y,z]
/// @param scale Scale parameters [sx,sy,sz]
/// @param R Precomputed rotation matrix from the quaternion
/// @param dLossDCovar Gradient of loss with respect to the covariance matrix
/// @return Tuple containing gradients for quaternion and scale parameters
template <typename T>
inline __device__ std::tuple<nanovdb::math::Vec4<T>, nanovdb::math::Vec3<T>>
quaternionAndScaleToCovarianceVectorJacobianProduct(const nanovdb::math::Vec4<T> &quat,
                                                    const nanovdb::math::Vec3<T> &scale,
                                                    // precompute
                                                    const nanovdb::math::Mat3<T> &R,
                                                    // grad outputs
                                                    const nanovdb::math::Mat3<T> &dLossDCovar) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = scale[0], sy = scale[1], sz = scale[2];

    // M = R * S
    const nanovdb::math::Mat3<T> S(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    const nanovdb::math::Mat3<T> M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    const nanovdb::math::Mat3<T> dLossDM = (dLossDCovar + dLossDCovar.transpose()) * M;
    const nanovdb::math::Mat3<T> dLossDR = dLossDM * S.transpose();

    // grad for (quat, scale) from covar
    const nanovdb::math::Vec4<T> &dLossDQuat =
        quaternionToRotationMatrixVectorJacobianProduct<T>(quat, dLossDR);

    // Row-major dot products for gradients
    const nanovdb::math::Vec3<T> dLossDScale(
        R[0][0] * dLossDM[0][0] + R[1][0] * dLossDM[1][0] + R[2][0] * dLossDM[2][0],
        R[0][1] * dLossDM[0][1] + R[1][1] * dLossDM[1][1] + R[2][1] * dLossDM[2][1],
        R[0][2] * dLossDM[0][2] + R[1][2] * dLossDM[1][2] + R[2][2] * dLossDM[2][2]);

    return { dLossDQuat, dLossDScale };
}

/// @brief Convert a quaternion and scale to a covariance matrix
///
/// This function computes a 3x3 covariance matrix from a quaternion and scale parameters.
/// The covariance matrix represents the shape and orientation of a 3D Gaussian distribution.
///
/// The computation follows the formula C = M * M^T where M = R * S, with:
/// - R being the rotation matrix derived from the quaternion
/// - S being the diagonal scale matrix
///
/// This representation allows for efficient transformation of Gaussian distributions
/// in 3D space, where the quaternion controls the orientation and the scale parameters
/// control the extent along each principal axis.
///
/// @param quat Input quaternion [w,x,y,z] representing rotation
/// @param scale Scale parameters [sx,sy,sz] representing extent along principal axes
/// @return 3x3 covariance matrix representing the Gaussian's shape and orientation
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
                  quaternionAndScaleToCovariance(const nanovdb::math::Vec4<T> &quat,
                                                 const nanovdb::math::Vec3<T> &scale) {
    const nanovdb::math::Mat3<T> &R = quaternionToRotationMatrix<T>(quat);
    // C = R * S * S * Rt
    const nanovdb::math::Mat3<T> S(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
    const nanovdb::math::Mat3<T> M = R * S;
    return M * M.transpose();
}

/// @brief Adds blur to a 2D covariance matrix and computes compensation factor
///
/// This function adds a small epsilon value to the diagonal elements of a 2D covariance matrix
/// to ensure numerical stability and prevent degenerate cases. It also computes a compensation
/// factor that can be used to adjust other calculations to account for this added blur.
///
/// The blur is added by increasing the diagonal elements of the covariance matrix by eps2d.
/// The compensation factor is calculated as the square root of the ratio between the original
/// determinant and the determinant after adding blur, which helps maintain proper normalization
/// of the Gaussian when rendered.
///
/// @param eps2d Epsilon value to add to diagonal elements of the covariance matrix
/// @param outCovar Input/output 2D covariance matrix that will be modified with added blur
/// @param outCompensation Output compensation factor to adjust for the added blur
/// @return Determinant of the covariance matrix after adding blur
template <typename T>
inline __device__ T
addBlur(const T eps2d, nanovdb::math::Mat2<T> &outCovar, T &outCompensation) {
    const T det_orig = outCovar[0][0] * outCovar[1][1] - outCovar[0][1] * outCovar[1][0];
    outCovar[0][0] += eps2d;
    outCovar[1][1] += eps2d;
    const T det_blur = outCovar[0][0] * outCovar[1][1] - outCovar[0][1] * outCovar[1][0];
    outCompensation  = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

/// @brief Computes the gradient of loss with respect to a covariance matrix from blur operations
///
/// This function implements the vector-Jacobian product calculation for the backward pass
/// of the addBlur operation. It propagates gradients from the compensation factor back
/// to the original covariance matrix before blur was applied.
///
/// During the forward pass, a blur is added to the covariance matrix by adding an epsilon
/// to its diagonal elements, and a compensation factor is computed to normalize the Gaussian.
/// This function computes how changes in that compensation factor affect the original matrix.
///
/// The calculations account for:
/// 1. The change in determinant from adding blur
/// 2. The relationship between the compensation factor and the covariance determinant
/// 3. The effect of the epsilon value on the gradient
///
/// @param eps2d Epsilon value that was added to the diagonal elements
/// @param conic_blur The 2x2 covariance matrix after blur was applied
/// @param compensation The compensation factor computed during forward pass
/// @param dLossDCompensation Gradient of loss with respect to the compensation factor
/// @return 2x2 matrix representing the gradient of loss with respect to the original covariance
template <typename T>
inline __device__ nanovdb::math::Mat2<T>
generateBlurVectorJacobianProduct(const T eps2d, const nanovdb::math::Mat2<T> &conic_blur,
                                  const T compensation, const T dLossDCompensation) {
    const T det_conic_blur =
        conic_blur[0][0] * conic_blur[1][1] - conic_blur[0][1] * conic_blur[1][0];
    const T v_sqr_comp         = dLossDCompensation * 0.5 / (compensation + 1e-6);
    const T one_minus_sqr_comp = 1 - compensation * compensation;
    return v_sqr_comp *
           nanovdb::math::Mat2<T>(one_minus_sqr_comp * conic_blur[0][0] - eps2d * det_conic_blur,
                                  one_minus_sqr_comp * conic_blur[0][1],
                                  one_minus_sqr_comp * conic_blur[1][0],
                                  one_minus_sqr_comp * conic_blur[1][1] - eps2d * det_conic_blur);
}

/// @brief Transform a point from world to camera coordinates
///
/// This function applies the world-to-camera transformation to convert a point
/// from world coordinates to camera coordinates. The transformation consists of:
/// - A rotation matrix that defines the camera's orientation in world space
/// - A translation vector that defines the camera's position in world space
///
/// @param camToWorldRotation Rotation matrix from camera to world
/// @param camToWorldTranslation Translation vector from camera to world
/// @param worldSpacePoint The point in world coordinates
/// @return The transformed point in camera coordinates
template <typename T>
inline __device__ nanovdb::math::Vec3<T>
                  transformPointWorldToCam(nanovdb::math::Mat3<T> const &camToWorldRotation,
                                           nanovdb::math::Vec3<T> const &camToWorldTranslation,
                                           nanovdb::math::Vec3<T> const &worldSpacePoint) {
    return camToWorldRotation * worldSpacePoint + camToWorldTranslation;
}

/// @brief Computes gradients for the world-to-camera point transformation
///
/// This function calculates the vector-Jacobian product (VJP) for the backward pass
/// of the point transformation from world to camera space. It propagates gradients
/// through the transformation p_camera = R * p_world + t.
///
/// Given upstream gradients with respect to the camera-space point (dL/dp_camera),
/// this function computes the gradients with respect to:
/// 1. The rotation matrix (dL/dR)
/// 2. The translation vector (dL/dt)
/// 3. The original world-space point (dL/dp_world)
///
/// The implementation follows the chain rule for matrix-vector operations:
/// - dL/dR = (dL/dp_camera) * (p_world)^T
/// - dL/dt = dL/dp_camera
/// - dL/dp_world = R^T * (dL/dp_camera)
///
/// @param camToWorldRotation Rotation matrix from camera to world
/// @param camToWorldTranslation Translation vector from camera to world
/// @param worldSpacePoint The original point in world coordinates
/// @param dLossDPointCamera Upstream gradient with respect to camera-space point (dL/dp_camera)
/// @return Tuple of gradients (dL/dR, dL/dt, dL/dp_world)
template <typename T>
inline __device__ std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Vec3<T>, nanovdb::math::Vec3<T>>
transformPointWorldToCamVectorJacobianProduct(const nanovdb::math::Mat3<T> &camToWorldRotation,
                                              const nanovdb::math::Vec3<T> &camToWorldTranslation,
                                              const nanovdb::math::Vec3<T> &worldSpacePoint,
                                              // grad
                                              const nanovdb::math::Vec3<T> &dLossDPointCamera) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    return { dLossDPointCamera.outer(worldSpacePoint), dLossDPointCamera,
             camToWorldRotation.transpose() * dLossDPointCamera };
}

/// @brief Transform a covariance matrix from world to camera coordinates
///
/// This function transforms a 3x3 covariance matrix from world coordinate space to
/// camera coordinate space using a rotation matrix. The transformation follows
/// the sandwich rule for covariance matrices:
///
/// covar_camera = R * covar_world * R^T
///
/// where R is the rotation matrix from world to camera space and R^T is its transpose.
///
/// This transformation preserves the properties of the covariance matrix while
/// correctly reorienting it according to the camera's viewpoint. It's commonly
/// used when projecting 3D Gaussian distributions to camera space for rendering
/// or further processing.
///
/// @param R Rotation matrix from world to camera space
/// @param covar Covariance matrix in world coordinates
/// @return Transformed covariance matrix in camera coordinates
template <typename T>
inline __device__ nanovdb::math::Mat3<T>
                  transformCovarianceWorldToCam(nanovdb::math::Mat3<T> const &R,
                                                nanovdb::math::Mat3<T> const &covar) {
    return R * covar * R.transpose();
}

/// @brief Computes gradients for the world-to-camera covariance transformation
///
/// This function calculates the vector-Jacobian product (VJP) for the backward pass
/// of the covariance matrix transformation from world to camera space. It propagates
/// gradients through the transformation covar_camera = R * covar_world * R^T.
///
/// Given upstream gradients with respect to the camera-space covariance (dL/dcovar_camera),
/// this function computes the gradients with respect to:
/// 1. The rotation matrix (dL/dR)
/// 2. The original world-space covariance matrix (dL/dcovar_world)
///
/// The implementation applies the chain rule for matrix operations:
/// - dL/dR = dL/dcovar_camera * R * covar_world^T + (dL/dcovar_camera)^T * R * covar_world
/// - dL/dcovar_world = R^T * dL/dcovar_camera * R
///
/// @param R Rotation matrix from world to camera
/// @param covar World-space covariance matrix
/// @param dLossDCovarCamera Upstream gradient with respect to camera-space covariance
/// @return Tuple of gradients (dL/dR, dL/dcovar_world)
template <typename T>
inline __device__ std::tuple<nanovdb::math::Mat3<T>, nanovdb::math::Mat3<T>>
                  transformCovarianceWorldToCamVectorJacobianProduct(
                      // fwd inputs
    const nanovdb::math::Mat3<T> &R, const nanovdb::math::Mat3<T> &covar,
    // grad outputs
    const nanovdb::math::Mat3<T> &dLossDCovarCamera) {
    // for D = W * X * WT, G = df/dD
    // df/dX = WT * G * W
    // df/dW
    // = G * (X * WT)T + ((W * X)T * G)T
    // = G * W * XT + (XT * WT * G)T
    // = G * W * XT + GT * W * X
    return { dLossDCovarCamera * R * covar.transpose() + dLossDCovarCamera.transpose() * R * covar,
             R.transpose() * dLossDCovarCamera * R };
}

/// @brief Computes gradient for matrix inverse operation
///
/// This function calculates the vector-Jacobian product (VJP) for the backward pass
/// of a matrix inverse operation. Given the inverse of a matrix (P = M^-1) and
/// the upstream gradient with respect to that inverse (dL/dP), it computes the
/// gradient with respect to the original matrix (dL/dM).
///
/// The mathematical formula used is:
/// dL/dM = -P * dL/dP * P = -M^-1 * dL/dP * M^-1
///
/// This calculation is derived from the differential of matrix inverse:
/// d(M^-1) = -M^-1 * dM * M^-1
///
/// @tparam T Matrix type that supports multiplication operations
/// @param MInv The inverse matrix (M^-1)
/// @param dLossDMInv Upstream gradient with respect to the inverse matrix (dL/dP)
/// @return Gradient with respect to the original matrix (dL/dM)
template <typename T>
inline __device__ T
inverseVectorJacobianProduct(const T &MInv, const T &dLossDMInv) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    return -MInv * dLossDMInv * MInv;
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANUTILS_CUH
