// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_OPS_H
#define FVDB_DETAIL_OPS_OPS_H

#include <Types.h>
#include <detail/GridBatchImpl.h>
#include <detail/utils/Utils.h>

#include <torch/extension.h>

namespace fvdb {
namespace detail {
namespace ops {

template <c10::DeviceType>
JaggedTensor dispatchJaggedTensorIndexInt(const JaggedTensor &jt, int64_t idxVal);

template <c10::DeviceType>
JaggedTensor
dispatchJaggedTensorIndexSlice(const JaggedTensor &jt, int64_t start, int64_t end, int64_t step);

template <c10::DeviceType>
JaggedTensor dispatchJaggedTensorIndexJaggedTensor(const JaggedTensor &jt, const JaggedTensor &idx);

template <c10::DeviceType> JaggedTensor dispatchJCat0(const std::vector<JaggedTensor> &tensors);

template <c10::DeviceType>
torch::Tensor dispatchJOffsetsForJIdx(torch::Tensor jidx, torch::Tensor jdata, int64_t numTensors);

template <c10::DeviceType>
torch::Tensor dispatchJIdxForJOffsets(torch::Tensor joffsets, int64_t numElements);

template <c10::DeviceType>
JaggedTensor dispatchEnabledMask(const GridBatchImpl &batchHdl, bool returnDisabled);

template <c10::DeviceType>
torch::Tensor dispatchJIdxForGrid(const GridBatchImpl &batchHdl, bool ignoreDisabledVoxels);

template <c10::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer> dispatchCreateNanoGridFromIJK(const JaggedTensor &ijk,
                                                                     bool isMutable);

template <c10::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromDense(uint32_t batchSize,
                                nanovdb::Coord origin,
                                nanovdb::Coord size,
                                bool isMutable,
                                torch::Device device,
                                const std::optional<torch::Tensor> &maybeMask);
template <c10::DeviceType>
void dispatchPopulateGridMetadata(const nanovdb::GridHandle<TorchDeviceBuffer> &batchHdl,
                                  const std::vector<nanovdb::Vec3d> &voxelSizes,
                                  const std::vector<nanovdb::Vec3d> &voxelOrigins,
                                  const bool isMutable,
                                  torch::Tensor &outBatchOffsets,
                                  GridBatchImpl::GridMetadata *outPerGridMetadataHost,
                                  GridBatchImpl::GridMetadata *outPerGridMetadataDevice,
                                  GridBatchImpl::GridBatchMetadata *outBatchMetadataHost,
                                  GridBatchImpl::GridBatchMetadata *outBatchMetadataDevice);

template <c10::DeviceType>
void dispatchReadIntoDense(const GridBatchImpl &batchHdl,
                           const torch::Tensor &inGridData,
                           const torch::Tensor &denseOrigins,
                           torch::Tensor &outDenseTensor,
                           bool ignoreMasked);

template <c10::DeviceType>
void dispatchReadFromDense(const GridBatchImpl &batchHdl,
                           const torch::Tensor &inDenseTensor,
                           const torch::Tensor &denseOrigins,
                           torch::Tensor &outSparseTensor,
                           bool ignoreMasked);

template <c10::DeviceType>
void dispatchFillFromGrid(const GridBatchImpl &fromGrid,
                          const GridBatchImpl &toGrid,
                          const torch::Tensor &fromFeatures,
                          torch::Tensor &toFeatures);

template <c10::DeviceType>
JaggedTensor
dispatchIjkToInvIndex(const GridBatchImpl &batchHdl, const JaggedTensor &ijk, bool cumulative);

template <c10::DeviceType>
JaggedTensor dispatchRayImplicitIntersection(const GridBatchImpl &batchHdl,
                                             const JaggedTensor &rayOrigins,
                                             const JaggedTensor &rayDirections,
                                             const JaggedTensor &gridScalars,
                                             float eps);

template <c10::DeviceType>
JaggedTensor
dispatchCoordsInGrid(const GridBatchImpl &batchHdl, const JaggedTensor &coords, bool ignoreMasked);

template <c10::DeviceType>
int64_t dispatchCountEnabledVoxels(const GridBatchImpl &batchHdl, int batchIdx = -1);

template <c10::DeviceType>
JaggedTensor dispatchActiveVoxelsInBoundsMask(const GridBatchImpl &batchHdl,
                                              const Vec3iBatch &ijkMin,
                                              const Vec3iBatch &ijkMax,
                                              bool ignoreDisabledVoxels);

template <c10::DeviceType>
void
dispatchSetMaskedIjk(const GridBatchImpl &batchHdl, const JaggedTensor &coords, bool maskedState);

template <c10::DeviceType>
std::vector<JaggedTensor> dispatchGridEdgeNetwork(const GridBatchImpl &gridHdl,
                                                  bool returnVoxelCoordinates);

template <c10::DeviceType>
JaggedTensor dispatchUniformRaySamples(const GridBatchImpl &batchHdl,
                                       const JaggedTensor &rayO,
                                       const JaggedTensor &rayD,
                                       const JaggedTensor &tMin,
                                       const JaggedTensor &tMax,
                                       const double minStepSize,
                                       const double coneAngle,
                                       const bool includeEndSegments,
                                       const bool return_midpoint,
                                       const double eps);

template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridMaxPoolBackward(const GridBatchImpl &coarseBatchHdl,
                                                    const GridBatchImpl &fineBatchHdl,
                                                    const torch::Tensor &fineData,
                                                    const torch::Tensor &coarseGradOut,
                                                    nanovdb::Coord poolingFactor,
                                                    nanovdb::Coord stride);

template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridMaxPool(const GridBatchImpl &fineBatchHdl,
                                            const GridBatchImpl &coarseBatchHdl,
                                            const torch::Tensor &fineData,
                                            nanovdb::Coord poolingFactor,
                                            nanovdb::Coord stride);

template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridAvgPoolBackward(const GridBatchImpl &coarseBatchHdl,
                                                    const GridBatchImpl &fineBatchHdl,
                                                    const torch::Tensor &fineData,
                                                    const torch::Tensor &coarseGradOut,
                                                    nanovdb::Coord poolingFactor,
                                                    nanovdb::Coord stride);

template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridAvgPool(const GridBatchImpl &fineBatchHdl,
                                            const GridBatchImpl &coarseBatchHdl,
                                            const torch::Tensor &fineData,
                                            nanovdb::Coord poolingFactor,
                                            nanovdb::Coord stride);

template <c10::DeviceType>
torch::Tensor dispatchUpsampleGridNearest(const GridBatchImpl &coarseBatchHdl,
                                          const GridBatchImpl &fineBatchHdl,
                                          const torch::Tensor &coarseData,
                                          nanovdb::Coord upsamplingFactor);

template <c10::DeviceType>
torch::Tensor dispatchUpsampleGridNearestBackward(const GridBatchImpl &fineBatchHdl,
                                                  const GridBatchImpl &coarseBatchHdl,
                                                  const torch::Tensor &gradOut,
                                                  const torch::Tensor &coarseData,
                                                  nanovdb::Coord upsamplingFactor);

template <c10::DeviceType>
JaggedTensor dispatchVoxelNeighborhood(const GridBatchImpl &batchHdl,
                                       const JaggedTensor &coords,
                                       nanovdb::Coord extentMin,
                                       nanovdb::Coord extentMax,
                                       int32_t shift);

template <c10::DeviceType>
JaggedTensor
dispatchIjkToIndex(const GridBatchImpl &batchHdl, const JaggedTensor &ijk, bool cumulative);

template <c10::DeviceType>
JaggedTensor
dispatchPointsInGrid(const GridBatchImpl &batchHdl, const JaggedTensor &points, bool ignoreMasked);

template <c10::DeviceType>
JaggedTensor dispatchCubesInGrid(const GridBatchImpl &batchHdl,
                                 const JaggedTensor &cubeCenters,
                                 const Vec3dOrScalar &padMin,
                                 const Vec3dOrScalar &padMax,
                                 bool ignoreDisabledVoxels);

template <c10::DeviceType>
JaggedTensor dispatchCubesIntersectGrid(const GridBatchImpl &batchHdl,
                                        const JaggedTensor &cubeCenters,
                                        const Vec3dOrScalar &padMin,
                                        const Vec3dOrScalar &padMax,
                                        bool ignoreDisabledVoxels);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridTrilinear(const GridBatchImpl &batchHdl,
                                                       const JaggedTensor &points,
                                                       const torch::Tensor &gridData);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridTrilinearWithGrad(const GridBatchImpl &batchHdl,
                                                               const JaggedTensor &points,
                                                               const torch::Tensor &gridData);

template <c10::DeviceType>
torch::Tensor dispatchSampleGridTrilinearWithGradBackward(const GridBatchImpl &batchHdl,
                                                          const JaggedTensor &points,
                                                          const torch::Tensor &data,
                                                          const torch::Tensor &gradOutFeatures,
                                                          const torch::Tensor &gradOutGradFeatures);

template <c10::DeviceType>
torch::Tensor dispatchSplatIntoGridTrilinear(const GridBatchImpl &batchHdl,
                                             const JaggedTensor &points,
                                             const torch::Tensor &gridData);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridBezier(const GridBatchImpl &batchHdl,
                                                    const JaggedTensor &points,
                                                    const torch::Tensor &gridData);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridBezierWithGrad(const GridBatchImpl &batchHdl,
                                                            const JaggedTensor &points,
                                                            const torch::Tensor &gridData);

template <c10::DeviceType>
torch::Tensor dispatchSampleGridBezierWithGradBackward(const GridBatchImpl &batchHdl,
                                                       const JaggedTensor &points,
                                                       const torch::Tensor &gradOutFeatures,
                                                       const torch::Tensor &gradOutGradFeatures,
                                                       const torch::Tensor &data);

template <c10::DeviceType>
torch::Tensor dispatchSplatIntoGridBezier(const GridBatchImpl &batchHdl,
                                          const JaggedTensor &points,
                                          const torch::Tensor &pointsData);

template <c10::DeviceType>
std::vector<JaggedTensor> dispatchVoxelsAlongRays(const GridBatchImpl &batchHdl,
                                                  const JaggedTensor &rayOrigins,
                                                  const JaggedTensor &rayDirections,
                                                  int64_t maxVox,
                                                  float eps,
                                                  bool returnIjk,
                                                  bool cumulative);

template <c10::DeviceType>
JaggedTensor dispatchSegmentsAlongRays(const GridBatchImpl &batchHdl,
                                       const JaggedTensor &rayOrigins,
                                       const JaggedTensor &rayDirections,
                                       int64_t maxSegments,
                                       const double eps,
                                       const bool ignoreMasked);

template <c10::DeviceType>
JaggedTensor dispatchActiveGridCoords(const GridBatchImpl &gridAccessor, bool ignoreDisabledVoxels);

template <c10::DeviceType>
torch::Tensor dispatchTransformPointsToGrid(const GridBatchImpl &batchHdl,
                                            const JaggedTensor &points,
                                            bool isPrimal);

template <c10::DeviceType>
torch::Tensor dispatchInvTransformPointsToGrid(const GridBatchImpl &batchHdl,
                                               const JaggedTensor &points,
                                               bool isPrimal);

template <c10::DeviceType>
torch::Tensor dispatchTransformPointsToGridBackward(const GridBatchImpl &batchHdl,
                                                    const JaggedTensor &gradOut,
                                                    bool isPrimal);

template <c10::DeviceType>
torch::Tensor dispatchInvTransformPointsToGridBackward(const GridBatchImpl &batchHdl,
                                                       const JaggedTensor &gradOut,
                                                       bool isPrimal);

template <c10::DeviceType>
void dispatchVolumeRender(const torch::Tensor sigmas,
                          const torch::Tensor rgbs,
                          const torch::Tensor deltas,
                          const torch::Tensor ts,
                          const torch::Tensor raysAcc,
                          const float opacityThreshold,
                          torch::Tensor &outOpacity,
                          torch::Tensor &outDepth,
                          torch::Tensor &outRgb,
                          torch::Tensor &outWs,
                          torch::Tensor &outTotalSamples);

template <c10::DeviceType>
std::vector<JaggedTensor>
dispatchMarchingCubes(const GridBatchImpl &batchHdl, const torch::Tensor &sdf, double level);

template <c10::DeviceType>
void dispatchVolumeRenderBackward(const torch::Tensor dLdOpacity,
                                  const torch::Tensor dLdDepth,
                                  const torch::Tensor dLdRgb,
                                  const torch::Tensor dLdWs,
                                  const torch::Tensor sigmas,
                                  const torch::Tensor rgbs,
                                  const torch::Tensor ws,
                                  const torch::Tensor deltas,
                                  const torch::Tensor ts,
                                  const torch::Tensor raysAcc,
                                  const torch::Tensor opacity,
                                  const torch::Tensor depth,
                                  const torch::Tensor rgb,
                                  const float opacityThreshold,
                                  torch::Tensor &outDLdSigmas,
                                  torch::Tensor &outDLdRbgs);

template <c10::DeviceType>
JaggedTensor dispatchIJKForMesh(const JaggedTensor &meshVertices,
                                const JaggedTensor &meshFaces,
                                const std::vector<VoxelCoordTransform> &transforms);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForGrid(const GridBatchImpl &batchHdl,
                                      const nanovdb::CoordBBox &bbox);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForGridWithoutBorder(const GridBatchImpl &batchHdl,
                                                   const nanovdb::CoordBBox &bbox);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForPoints(const JaggedTensor &points,
                                        const nanovdb::CoordBBox &bbox,
                                        const std::vector<VoxelCoordTransform> &transforms);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForCoords(const JaggedTensor &coords, const nanovdb::CoordBBox &bbox);

template <c10::DeviceType>
JaggedTensor
dispatchNearestNeighborIJKForPoints(const JaggedTensor &points,
                                    const std::vector<VoxelCoordTransform> &transforms);

template <c10::DeviceType>
JaggedTensor dispatchCoarseIJKForFineGrid(const GridBatchImpl &batchHdl,
                                          nanovdb::Coord coarseningFactor);

template <c10::DeviceType>
JaggedTensor dispatchFineIJKForCoarseGrid(const GridBatchImpl &batchHdl,
                                          nanovdb::Coord upsamplingFactor,
                                          const std::optional<JaggedTensor> &maybeMask);

template <c10::DeviceType>
JaggedTensor dispatchConvIJKForGrid(const GridBatchImpl &batchHdl,
                                    const nanovdb::Coord &kernelSize,
                                    const nanovdb::Coord &stride);

template <c10::DeviceType>
torch::Tensor dispatchScaledDotProductAttention(const torch::Tensor &query,
                                                const torch::Tensor &key,
                                                const torch::Tensor &value,
                                                const torch::Tensor &qLengths,
                                                const torch::Tensor &kvLengths,
                                                bool training,
                                                float scale);

// template <c10::DeviceType>
// std::tuple<torch::Tensor, torch::Tensor> dispatchQuatScaleToCovarPerciForward(
//     const torch::Tensor &quats,  // [N, 4]
//     const torch::Tensor &scales, // [N, 3]
//     const bool compute_covar,
//     const bool compute_preci,
//     const bool triu
// );

/// @defgroup ops_gsplat Gaussian Splatting Operations
/// @brief Operations for 3D Gaussian Splatting
///
/// Tensor Dimensions Legend:
/// - B: number of batches
/// - C: number of cameras
/// - N: number of points / Gaussians
/// - M: total Gaussians across all cameras (non-batched) or across all batches (batched).
///      M can also be thought of as the number of Gaussian-camera pairs.
/// - BC: total Cameras across all batches
/// - K: degree of the spherical harmonics
/// - D: number of feature/color channels
///
/// Gaussians splats are represented by:
/// - means: 3D positions of Gaussians [N, 3]
/// - quats: Quaternion rotations of Gaussians [N, 4]
/// - scales: Scale factors of Gaussians [N, 3]
///
/// The quats and scales define an ellipsoid (equivalently covariance) for each Gaussian.
///
/// Camera parameters are represented by:
/// - viewmats: Camera view matrices [C, 4, 4]
/// - Ks: Camera intrinsic (projection) matrices [C, 3, 3]
/// @{

/// @brief Evaluate spherical harmonics functions to compute features/colors.
///
/// This function computes the features/colors for points in 3D space using spherical harmonics
/// (SH) representation. Spherical harmonics provide an efficient way to represent view-dependent
/// appearance for Gaussian Splatting and other rendering techniques. The output features are not
/// limited to RGB colors; they can have any number of channels.
///
/// @param[in] shDegreeToUse Degree of spherical harmonics to use (0-3 typically, higher degrees
/// provide more detail)
/// @param[in] numCameras Number of cameras used for rendering
/// @param[in] viewDirs Direction vectors [N, 3] (packed) or [C, N, 3] (unpacked) normalized to unit
/// length, representing view directions
/// @param[in] shCoeffs Spherical harmonic coefficients [N, K, 3] (packed) or
/// [K, C, N, 3] (unpacked), where K depends on sh_degree_to_use (K=(sh_degree_to_use+1)²)
/// @param[in] radii radii [N] (packed) or [C, N] (unpacked) for view-dependent level-of-detail
/// control
///
/// @return Features/colors [N, D] computed from the spherical harmonics evaluation
template <c10::DeviceType>
torch::Tensor dispatchSphericalHarmonicsForward(const int64_t shDegreeToUse,
                                                const int64_t numCameras,
                                                const torch::Tensor &viewDirs,  // [C, N, 3]
                                                const torch::Tensor &sh0Coeffs, // [1, N, D]
                                                const torch::Tensor &shNCoeffs, // [N, K-1, D]
                                                const torch::Tensor &radii      // [C, N]
);

/// @brief Spherical harmonics evaluation backward pass
///
/// This function computes the vector-Jacobian product between the output gradients and the
/// Jacobian of the spherical harmonics forward operation.
///
/// @param[in] shDegreeToUse Degree of spherical harmonics used in the forward pass
/// @param[in] numCameras Number of cameras used in the forward pass
/// @param[in] numGaussians Number of Gaussians used in the forward pass
/// @param[in] viewDirs Direction vectors [N, 3] (packed) or [C, N, 3] (unpacked) used in the
/// forward pass
/// @param[in] shCoeffs Spherical harmonic coefficients [N, K, 3] (packed) or [K, C, N, 3]
/// (unpacked) where K depends on sh_degree_to_use
/// @param[in] dLossDColors Gradients of the loss function with respect to output colors [N, 3]
/// - ∂L/∂colors
/// @param[in] radii radii [N] (packed) or [C, N] (unpacked) used in the forward pass for
/// level-of-detail
/// @param[in] computeDLossDViewDirs Whether to compute gradients with respect to direction
/// vectors
///
/// @return std::tuple containing gradients of the loss function with respect to:
///         - SH coefficients [N, K, 3] - ∂L/∂sh_coeffs
///         - Direction vectors [N, 3] - ∂L/∂dirs (if compute_v_dirs is true, otherwise empty
///         tensor)
template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward(const int64_t shDegreeToUse,
                                   const int64_t numCameras,
                                   const int64_t numGaussians,
                                   const torch::Tensor &viewDirs,  // [N, 3]
                                   const torch::Tensor &shNCoeffs, // [N, K-1, D]
                                   const torch::Tensor &dLossDColors,
                                   const torch::Tensor &radii,     // [N]
                                   const bool computeDLossDViewDirs);

/// @brief Project 3D Gaussians to 2D screen space pixel coordinates for rendering
///
/// This function transforms 3D Gaussians to 2D screen space by applying camera projections.
/// It computes the 2D means, depths, 2D covariance matrices (conics), and potentially compensation
/// factors to accurately represent the 3D Gaussians in 2D for later rasterization.
///
/// The origin of the 2D pixel coordinates is the top-left corner of the image, with positive x-axis
/// pointing to the right and positive y-axis pointing downwards.
///
/// @attention The output radii of 3D Gaussians that are discarded (due to clipping or projection
/// too small) are set to zero, but the other output values of discarded Gaussians are uninitialized
/// (undefined).
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means 3D positions of Gaussians [N, 3] where N is number of Gaussians
/// @param[in] quats Quaternion rotations of Gaussians [N, 4] in format (x, y, z, w)
/// @param[in] scales Scale factors of Gaussians [N, 3] representing extent in each dimension
/// @param[in] worldToCamMatrices Camera view matrices [C, 4, 4] where C is number of cameras
/// @param[in] projectionMatrices Camera intrinsic matrices [C, 3, 3]
/// @param[in] imageWidth Width of the output image in pixels
/// @param[in] imageHeight Height of the output image in pixels
/// @param[in] eps2d 2D projection epsilon for numerical stability
/// @param[in] nearPlane Near clipping plane distance
/// @param[in] farPlane Far clipping plane distance
/// @param[in] minRadius2d Radius clipping value to limit the maximum size of projected Gaussians
/// @param[in] calcCompensations Whether to calculate view-dependent compensation factors
/// @param[in] ortho Whether to use orthographic projection instead of perspective
///
/// @return std::tuple containing:
///         - 2D projected Gaussian centers [C, N, 2]
///         - Depths of Gaussians [C, N]
///         - Covariance matrices in conic form [C, N, 3] representing (a, b, c) in ax² + 2bxy + cy²
///         - Radii of 2D Gaussians [C, N]
///         - Compensation factors [C, N] (if calc_compensations is true, otherwise empty tensor)
template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForward(const torch::Tensor &means,              // [N, 3]
                                  const torch::Tensor &quats,              // [N, 4]
                                  const torch::Tensor &scales,             // [N, 3]
                                  const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
                                  const torch::Tensor &projectionMatrices, // [C, 3, 3]
                                  const int64_t imageWidth,
                                  const int64_t imageHeight,
                                  const float eps2d,
                                  const float nearPlane,
                                  const float farPlane,
                                  const float minRadius2d,
                                  const bool calcCompensations,
                                  const bool ortho);

/// @brief Calculate gradients for the 3D to 2D Gaussian projection (backward pass)
///
/// This function computes the gradients of the 3D to 2D Gaussian projection with respect to
/// the input parameters: 3D means, quaternions, scales, view matrices, and optionally camera
/// intrinsics. It enables backpropagation through the projection step in the Gaussian Splatting
/// pipeline.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means 3D positions of Gaussians [N, 3]
/// @param[in] quats Quaternion rotations of Gaussians [N, 4] in format (x, y, z, w)
/// @param[in] scales Scale factors of Gaussians [N, 3] representing extent in each dimension
/// @param[in] worldToCamMatrices Camera view matrices [C, 4, 4]
/// @param[in] projectionMatrices Camera intrinsic matrices [C, 3, 3]
/// @param[in] compensations View-dependent compensation factors [N, 6] (optional)
/// @param[in] imageWidth Width of the image in pixels
/// @param[in] imageHeight Height of the image in pixels
/// @param[in] eps2d 2D projection epsilon for numerical stability
/// @param[in] radii Output radii from forward pass [C, N]
/// @param[in] conics Output conics from forward pass [C, N, 3]
/// @param[out] dLossDMeans2d Gradients with respect to projected 2D means [C, N, 2]
/// @param[out] dLossDDepths Gradients with respect to depths [C, N]
/// @param[out] dLossDConics Gradients with respect to conics [C, N, 3]
/// @param[out] dLossDCompensations Gradients with respect to compensations [C, N] (optional)
/// @param[in] worldToCamMatricesRequiresGrad Whether viewmats requires gradient
/// @param[in] ortho Whether orthographic projection was used in forward pass
/// @param[in] outNormalizeddLossdMeans2dNormAccum Optional output for normalized gradients tracked
/// across backward passes
/// @param[in] outNormalizedMaxRadiiAccum Optional output for maximum radii tracked across backward
/// passses
/// @param[in] outGradientStepCounts Optional output for the number of times each gradient was
/// counted tracked across backward passes
///
/// @return std::tuple containing gradients of the loss function with respect to the input
/// parameters:
///         - (empty tensor placeholder for compatibility with forward pass)
///         - 2D means [C, N, 2] - gradients ∂L/∂means2d
///         - conics [C, N, 3] - gradients ∂L/∂conics
///         - colors [C, N, D] - gradients ∂L/∂colors
///         - opacities [N] - gradients ∂L/∂opacities
template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionBackward(
    const torch::Tensor &means,                       // [N, 3]
    const torch::Tensor &quats,                       // [N, 4]
    const torch::Tensor &scales,                      // [N, 3]
    const torch::Tensor &worldToCamMatrices,          // [C, 4, 4]
    const torch::Tensor &projectionMatrices,          // [C, 3, 3]
    const at::optional<torch::Tensor> &compensations, // [N, 6] optional
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const float eps2d,
    const torch::Tensor &radii,                             // [C, N]
    const torch::Tensor &conics,                            // [C, N, 3]
    const torch::Tensor &dLossDMeans2d,                     // [C, N, 2]
    const torch::Tensor &dLossDDepths,                      // [C, N]
    const torch::Tensor &dLossDConics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &dLossDCompensations, // [C, N] optional
    const bool worldToCamMatricesRequiresGrad,
    const bool ortho,
    at::optional<torch::Tensor> outNormalizeddLossdMeans2dNormAccum = std::nullopt,
    at::optional<torch::Tensor> outNormalizedMaxRadiiAccum          = std::nullopt,
    at::optional<torch::Tensor> outGradientStepCounts               = std::nullopt);

/// @brief Compute the intersection of 2D Gaussians with image tiles for efficient rasterization
///
/// This function determines which Gaussians intersect with which tiles in the rendered image,
/// which is a critical optimization for the Gaussian Splatting rendering pipeline.
/// It accelerates rendering by ensuring that only relevant Gaussians are processed for each image
/// tile.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means2d 2D projected Gaussian centers [C, N, 2] or [M, 2]
/// @param[in] radii Radii of 2D Gaussians [C, N] or [M]
/// @param[in] depths Depths of Gaussians [C, N] or [M] used for occlusion handling
/// @param[in] cameraIds Camera IDs for each Gaussian [M] (optional, NULL if using [C, N, ...]
/// format)
/// @param[in] numCameras Number of cameras
/// @param[in] tileSize Size of each tile in pixels (typically 16x16)
/// @param[in] numTilesH Number of tiles in the vertical dimension
/// @param[in] numTilesW Number of tiles in the horizontal dimension
///
/// @return std::tuple containing:
///         - Tile offsets [C, num_tiles_h, num_tiles_w] indicating for each tile where its
///         Gaussians
///           start
///         - Flattened Gaussian IDs [n_isects] indicating which Gaussians affect each tile
template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersection(const torch::Tensor &means2d, // [C, N, 2] or [M, 2]
                                 const torch::Tensor &radii,   // [C, N] or [M]
                                 const torch::Tensor &depths,  // [C, N] or [M]
                                 const at::optional<torch::Tensor> &cameraIds, // NULL or [M]
                                 const uint32_t numCameras,
                                 const uint32_t tileSize,
                                 const uint32_t numTilesH,
                                 const uint32_t numTilesW);

/// @brief Perform Gaussian rasterization to render an image (forward pass)
///
/// This function rasterizes 2D Gaussians into an image using a tile-based approach for efficiency.
/// Each Gaussian is represented by its 2D projected center, covariance matrix in conic form,
/// feature/color, and opacity. The function performs alpha-blending of the Gaussians to generate
/// the final rendered image.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means2d 2D projected Gaussian centers [C, N, 2]
/// @param[in] conics Gaussian covariance matrices in conic form [C, N, 3] representing (a, b, c) in
/// ax² + 2bxy + cy²
/// @param[in] features Feature / color values of Gaussians [C, N, D]
/// @param[in] opacities Opacity values for each Gaussian [N]
/// @param[in] imageWidth Width of the output image in pixels
/// @param[in] imageHeightimageHeight Height of the output image in pixels
/// @param[in] imageOriginW X-coordinate of the image origin (left)
/// @param[in] imageOriginH Y-coordinate of the image origin (top)
/// @param[in] tileSize Size of tiles used for rasterization optimization
/// @param[in] tileOffsets Offsets for tiles [C, tile_height, tile_width] indicating for each tile
/// where its Gaussians start
/// @param[in] tileGaussianIds Flattened Gaussian IDs for tile intersection [n_isects] indicating
/// which Gaussians affect each tile
///
/// @return std::tuple containing:
///         - Rendered image features/colors [C, image_height, image_width, D]
///         - Alpha values [C, image_height, image_width, 1]
///         - Last Gaussian ID rendered at each pixel [C, image_height, image_width]
template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeForward(const torch::Tensor &means2d,   // [C, N, 2]
                                 const torch::Tensor &conics,    // [C, N, 3]
                                 const torch::Tensor &features,  // [C, N, D]
                                 const torch::Tensor &opacities, // [N]
                                 const uint32_t imageWidth,
                                 const uint32_t imageHeight,
                                 const uint32_t imageOriginW,
                                 const uint32_t imageOriginH,
                                 const uint32_t tileSize,
                                 const torch::Tensor &tileOffsets, // [C, tile_height, tile_width]
                                 const torch::Tensor &tileGaussianIds // [n_isects]
);

/// @brief Calculate gradients for the Gaussian rasterization process (backward pass)
///
/// This function computes the gradients of the Gaussian splatting rendering with respect to
/// its input parameters: 2D projected Gaussian means, conics, features/colors, and opacities.
/// It is used during backpropagation to update the Gaussian parameters during training.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means2d 2D projected Gaussian centers [C, N, 2]
/// @param[in] conics Gaussian covariance matrices in conic form [C, N, 3] representing (a, b, c) in
/// ax² + 2bxy + cy²
/// @param[in] features Feature / color values of Gaussians [C, N, D]
/// @param[in] opacities Opacity values for each Gaussian [N]
/// @param[in] imageWidth Width of the rendered image
/// @param[in] imageHeight Height of the rendered image
/// @param[in] imageOriginW X-coordinate of the image origin (left)
/// @param[in] imageOriginH Y-coordinate of the image origin (top)
/// @param[in] tileSize Size of tiles used for rasterization optimization
/// @param[in] tileOffsets Offsets for tiles [C, tile_height, tile_width]
/// @param[in] tileGaussianIds Flattened Gaussian IDs for tile intersection [n_isects]
/// @param[in] renderedAlphas Alpha values from forward pass [C, image_height, image_width, 1]
/// @param[in] lastIds Last Gaussian IDs per pixel from forward pass [C, image_height, image_width]
/// @param[out] dLossDRenderedFeatures Gradients of loss with respect to rendered features [C,
/// image_height, image_width, D]
/// @param[out] dLossDRenderedAlphas Gradients of loss with respect to rendered alphas [C,
/// image_height, image_width, 1]
/// @param[in] absgrad Whether to use absolute gradients
/// @param[in] numSharedChannelsOverride Override for number of shared memory channels (-1 means
/// auto-select)
///
/// @return std::tuple containing gradients of the loss function with respect to the input
/// parameters:
///         - (empty tensor placeholder for compatibility with forward pass)
///         - 2D means [C, N, 2] - gradients ∂L/∂means2d
///         - conics [C, N, 3] - gradients ∂L/∂conics
///         - features [C, N, D] - gradients ∂L/∂features
///         - opacities [N] - gradients ∂L/∂opacities
template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeBackward(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,            // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds,        // [n_isects]
    const torch::Tensor &renderedAlphas,         // [C, imageHeight, imageWidth, 1]
    const torch::Tensor &lastIds,                // [C, imageHeight, imageWidth]
    const torch::Tensor &dLossDRenderedFeatures, // [C, imageHeight, imageWidth, D]
    const torch::Tensor &dLossDRenderedAlphas,   // [C, imageHeight, imageWidth, 1]
    const bool absgrad,
    const int64_t numSharedChannelsOverride = -1);

/// @brief Project 3D Gaussians to 2D screen space using jagged tensors for batched processing
///
/// This function transforms batches of 3D Gaussians to 2D screen space by applying camera
/// projections. It handles jagged (variable-sized) inputs for efficient batch processing, where
/// each batch element may contain a different number of Gaussians and cameras.
///
/// @attention The output radii of 3D Gaussians that are discarded (due to clipping or projection
/// too small) are set to zero, but the other output values of discarded Gaussians are uninitialized
/// (undefined).
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] gSizes Batch sizes for Gaussians [B]
/// @param[in] means 3D positions of Gaussians [M, 3]
/// @param[in] quats Quaternion rotations of Gaussians [M, 4] in format (x, y, z, w)
/// @param[in] scales Scale factors of Gaussians [M, 3] representing extent in each dimension
/// @param[in] cSizes Batch sizes for cameras [B]
/// @param[in] worldToCamMatrices Camera view matrices [BC, 4, 4]
/// @param[in] projectionMatrices Camera intrinsic matrices [BC, 3, 3]
/// @param[in] imageWidth Width of the output image in pixels
/// @param[in] imageHeight Height of the output image in pixels
/// @param[in] eps2d 2D projection epsilon for numerical stability
/// @param[in] nearPlane Near clipping plane distance
/// @param[in] farPlane Far clipping plane distance
/// @param[in] minRadius2d Radius clipping value to limit the maximum size of projected Gaussians
/// @param[in] ortho Whether to use orthographic projection instead of perspective
///
/// @return std::tuple containing:
///         - 2D projected Gaussian centers [M, 2]
///         - Depths of Gaussians [M]
///         - Covariance matrices in conic form [M, 3] representing (a, b, c) in ax² + 2bxy + cy²
///         - Radii of 2D Gaussians [M]
///         - Flattened camera indices [M] indicating which camera each projection corresponds to
template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedForward(const torch::Tensor &gSizes, // [B] gaussian sizes
                                        const torch::Tensor &means,  // [N, 3]
                                        const torch::Tensor &quats,  // [N, 4] optional
                                        const torch::Tensor &scales, // [N, 3] optional
                                        const torch::Tensor &cSizes, // [B] camera sizes
                                        const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
                                        const torch::Tensor &projectionMatrices, // [C, 3, 3]
                                        const uint32_t imageWidth,
                                        const uint32_t imageHeight,
                                        const float eps2d,
                                        const float nearPlane,
                                        const float farPlane,
                                        const float minRadius2d,
                                        const bool ortho);

/// @brief Calculate gradients for the jagged 3D to 2D Gaussian projection (backward pass)
///
/// This function computes the gradients of the 3D to 2D Gaussian projection with respect to
/// the input parameters when using jagged tensors for batch processing. It enables backpropagation
/// through the projection step in the Gaussian Splatting pipeline for scenes with variable
/// numbers of objects and cameras per batch.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] gSizes Batch sizes for Gaussians [B]
/// @param[in] means 3D positions of Gaussians [M, 3]
/// @param[in] quats Quaternion rotations of Gaussians [M, 4] in format (x, y, z, w)
/// @param[in] scales Scale factors of Gaussians [M, 3] representing extent in each dimension
/// @param[in] cSizes Batch sizes for cameras [B]
/// @param[in] worldToCamMatrices Camera view matrices [BC, 4, 4]
/// @param[in] projectionMatrices Camera intrinsic matrices [BC, 3, 3]
/// @param[in] imageWidth Width of the output image in pixels
/// @param[in] imageHeight Height of the output image in pixels
/// @param[in] eps2d 2D projection epsilon for numerical stability
/// @param[in] radii Output radii from forward pass [M]
/// @param[in] conics Output conics from forward pass [M, 3]
/// @param[out] dLossDMeans2d Gradients with respect to projected 2D means [M, 2]
/// @param[out] dLossDDepths Gradients with respect to depths [M]
/// @param[out] dLossDConics Gradients with respect to conics [M, 3]
/// @param[in] worldToCamMatricesRequiresGrad Whether viewmats requires gradient
/// @param[in] ortho Whether orthographic projection was used in forward pass
///
/// @return std::tuple containing gradients of the loss function with respect to the input
/// parameters:
///         - 3D means [M, 3] - ∂L/∂means
///         - Quaternions [M, 4] - ∂L/∂quats
///         - Scales [M, 3] - ∂L/∂scales
///         - View matrices [BC, 4, 4] - ∂L/∂viewmats (if viewmats_requires_grad is true, otherwise
/// empty tensor)
///         - Camera intrinsics [BC, 3, 3] - ∂L/∂Ks
template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedBackward(const torch::Tensor &gSizes, // [B] gaussian sizes
                                         const torch::Tensor &means,  // [N, 3]
                                         const torch::Tensor &quats,  // [N, 4] optional
                                         const torch::Tensor &scales, // [N, 3] optional
                                         const torch::Tensor &cSizes, // [B] camera sizes
                                         const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
                                         const torch::Tensor &projectionMatrices, // [C, 3, 3]
                                         const uint32_t imageWidth,
                                         const uint32_t imageHeight,
                                         const float eps2d,
                                         const torch::Tensor &radii,         // [N]
                                         const torch::Tensor &conics,        // [N, 3]
                                         const torch::Tensor &dLossDMeans2d, // [N, 2]
                                         const torch::Tensor &dLossDDepths,  // [N]
                                         const torch::Tensor &dLossDConics,  // [N, 3]
                                         const bool worldToCamMatricesRequiresGrad,
                                         const bool ortho);

/// @brief Create a mask identifying NaN or Inf values in Gaussian parameters
///
/// This function examines jagged tensors containing Gaussian parameters and creates a mask
/// that identifies any NaN (Not a Number) or Inf (Infinity) values. This is important for
/// numerical stability in Gaussian Splatting algorithms, allowing invalid Gaussians to be
/// filtered out before rendering.
///
/// @tparam DeviceType Device type template parameter (torch::kCUDA or torch::kCPU)
///
/// @param[in] means 3D positions of Gaussians as a jagged tensor [C, N, 3]
/// @param[in] quats Quaternion rotations of Gaussians as a jagged tensor [C, N, 4]
/// @param[in] scales Scale factors of Gaussians as a jagged tensor [C, N, 3]
/// @param[in] opacities Opacity values of Gaussians as a jagged tensor [N]
/// @param[in] sh0 Constant term (degree 0) spherical harmonic coefficients as a jagged tensor
/// @param[in] shN Higher degree spherical harmonic coefficients as a jagged tensor
///
/// @return A jagged tensor mask where True indicates valid values (no NaN/Inf) and False indicates
/// invalid values
template <c10::DeviceType>
fvdb::JaggedTensor dispatchGaussianNanInfMask(const fvdb::JaggedTensor &means,
                                              const fvdb::JaggedTensor &quats,
                                              const fvdb::JaggedTensor &logScales,
                                              const fvdb::JaggedTensor &logitOpacities,
                                              const fvdb::JaggedTensor &sh0,
                                              const fvdb::JaggedTensor &shN);

/// @} // end of ops_gsplat doxygen group

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_OPS_H
