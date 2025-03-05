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
JaggedTensor dispatchJaggedTensorIndexSlice(const JaggedTensor &jt, int64_t start, int64_t end,
                                            int64_t step);

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
                                                                     bool                isMutable);

template <c10::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromDense(uint32_t batchSize, nanovdb::Coord origin, nanovdb::Coord size,
                                bool isMutable, torch::Device device,
                                const torch::optional<torch::Tensor> &maybeMask);
template <c10::DeviceType>
void dispatchPopulateGridMetadata(const nanovdb::GridHandle<TorchDeviceBuffer> &batchHdl,
                                  const std::vector<nanovdb::Vec3d>            &voxelSizes,
                                  const std::vector<nanovdb::Vec3d>            &voxelOrigins,
                                  const bool isMutable, torch::Tensor &outBatchOffsets,
                                  GridBatchImpl::GridMetadata      *outPerGridMetadataHost,
                                  GridBatchImpl::GridMetadata      *outPerGridMetadataDevice,
                                  GridBatchImpl::GridBatchMetadata *outBatchMetadataHost,
                                  GridBatchImpl::GridBatchMetadata *outBatchMetadataDevice);

template <c10::DeviceType>
void dispatchReadIntoDense(const GridBatchImpl &batchHdl, const torch::Tensor &inGridData,
                           const torch::Tensor &denseOrigins, torch::Tensor &outDenseTensor,
                           bool ignoreMasked);

template <c10::DeviceType>
void dispatchReadFromDense(const GridBatchImpl &batchHdl, const torch::Tensor &inDenseTensor,
                           const torch::Tensor &denseOrigins, torch::Tensor &outSparseTensor,
                           bool ignoreMasked);

template <c10::DeviceType>
void dispatchFillFromGrid(const GridBatchImpl &fromGrid, const GridBatchImpl &toGrid,
                          const torch::Tensor &fromFeatures, torch::Tensor &toFeatures);

template <c10::DeviceType>
JaggedTensor dispatchIjkToInvIndex(const GridBatchImpl &batchHdl, const JaggedTensor &ijk,
                                   bool cumulative);

template <c10::DeviceType>
JaggedTensor dispatchRayImplicitIntersection(const GridBatchImpl &batchHdl,
                                             const JaggedTensor  &rayOrigins,
                                             const JaggedTensor  &rayDirections,
                                             const JaggedTensor &gridScalars, float eps);

template <c10::DeviceType>
JaggedTensor dispatchCoordsInGrid(const GridBatchImpl &batchHdl, const JaggedTensor &coords,
                                  bool ignoreMasked);

template <c10::DeviceType>
int64_t dispatchCountEnabledVoxels(const GridBatchImpl &batchHdl, int batchIdx = -1);

template <c10::DeviceType>
JaggedTensor dispatchActiveVoxelsInBoundsMask(const GridBatchImpl &batchHdl,
                                              const Vec3iBatch &ijkMin, const Vec3iBatch &ijkMax,
                                              bool ignoreDisabledVoxels);

template <c10::DeviceType>
void dispatchSetMaskedIjk(const GridBatchImpl &batchHdl, const JaggedTensor &coords,
                          bool maskedState);

template <c10::DeviceType>
std::vector<JaggedTensor> dispatchGridEdgeNetwork(const GridBatchImpl &gridHdl,
                                                  bool                 returnVoxelCoordinates);

template <c10::DeviceType>
JaggedTensor dispatchUniformRaySamples(const GridBatchImpl &batchHdl, const JaggedTensor &rayO,
                                       const JaggedTensor &rayD, const JaggedTensor &tMin,
                                       const JaggedTensor &tMax, const double minStepSize,
                                       const double coneAngle, const bool includeEndSegments,
                                       const bool return_midpoint, const double eps);

template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridMaxPoolBackward(const GridBatchImpl &coarseBatchHdl,
                                                    const GridBatchImpl &fineBatchHdl,
                                                    const torch::Tensor &fineData,
                                                    const torch::Tensor &coarseGradOut,
                                                    nanovdb::Coord       poolingFactor,
                                                    nanovdb::Coord       stride);

template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridMaxPool(const GridBatchImpl &fineBatchHdl,
                                            const GridBatchImpl &coarseBatchHdl,
                                            const torch::Tensor &fineData,
                                            nanovdb::Coord poolingFactor, nanovdb::Coord stride);

template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridAvgPoolBackward(const GridBatchImpl &coarseBatchHdl,
                                                    const GridBatchImpl &fineBatchHdl,
                                                    const torch::Tensor &fineData,
                                                    const torch::Tensor &coarseGradOut,
                                                    nanovdb::Coord       poolingFactor,
                                                    nanovdb::Coord       stride);

template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridAvgPool(const GridBatchImpl &fineBatchHdl,
                                            const GridBatchImpl &coarseBatchHdl,
                                            const torch::Tensor &fineData,
                                            nanovdb::Coord poolingFactor, nanovdb::Coord stride);

template <c10::DeviceType>
torch::Tensor
dispatchUpsampleGridNearest(const GridBatchImpl &coarseBatchHdl, const GridBatchImpl &fineBatchHdl,
                            const torch::Tensor &coarseData, nanovdb::Coord upsamplingFactor);

template <c10::DeviceType>
torch::Tensor dispatchUpsampleGridNearestBackward(const GridBatchImpl &fineBatchHdl,
                                                  const GridBatchImpl &coarseBatchHdl,
                                                  const torch::Tensor &gradOut,
                                                  const torch::Tensor &coarseData,
                                                  nanovdb::Coord       upsamplingFactor);

template <c10::DeviceType>
JaggedTensor dispatchVoxelNeighborhood(const GridBatchImpl &batchHdl, const JaggedTensor &coords,
                                       nanovdb::Coord extentMin, nanovdb::Coord extentMax,
                                       int32_t shift);

template <c10::DeviceType>
JaggedTensor dispatchIjkToIndex(const GridBatchImpl &batchHdl, const JaggedTensor &ijk,
                                bool cumulative);

template <c10::DeviceType>
JaggedTensor dispatchPointsInGrid(const GridBatchImpl &batchHdl, const JaggedTensor &points,
                                  bool ignoreMasked);

template <c10::DeviceType>
JaggedTensor dispatchCubesInGrid(const GridBatchImpl &batchHdl, const JaggedTensor &cubeCenters,
                                 const Vec3dOrScalar &padMin, const Vec3dOrScalar &padMax,
                                 bool ignoreDisabledVoxels);

template <c10::DeviceType>
JaggedTensor dispatchCubesIntersectGrid(const GridBatchImpl &batchHdl,
                                        const JaggedTensor  &cubeCenters,
                                        const Vec3dOrScalar &padMin, const Vec3dOrScalar &padMax,
                                        bool ignoreDisabledVoxels);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridTrilinear(const GridBatchImpl &batchHdl,
                                                       const JaggedTensor  &points,
                                                       const torch::Tensor &gridData);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridTrilinearWithGrad(const GridBatchImpl &batchHdl,
                                                               const JaggedTensor  &points,
                                                               const torch::Tensor &gridData);

template <c10::DeviceType>
torch::Tensor dispatchSampleGridTrilinearWithGradBackward(const GridBatchImpl &batchHdl,
                                                          const JaggedTensor  &points,
                                                          const torch::Tensor &data,
                                                          const torch::Tensor &gradOutFeatures,
                                                          const torch::Tensor &gradOutGradFeatures);

template <c10::DeviceType>
torch::Tensor dispatchSplatIntoGridTrilinear(const GridBatchImpl &batchHdl,
                                             const JaggedTensor  &points,
                                             const torch::Tensor &gridData);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridBezier(const GridBatchImpl &batchHdl,
                                                    const JaggedTensor  &points,
                                                    const torch::Tensor &gridData);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridBezierWithGrad(const GridBatchImpl &batchHdl,
                                                            const JaggedTensor  &points,
                                                            const torch::Tensor &gridData);

template <c10::DeviceType>
torch::Tensor dispatchSampleGridBezierWithGradBackward(const GridBatchImpl &batchHdl,
                                                       const JaggedTensor  &points,
                                                       const torch::Tensor &gradOutFeatures,
                                                       const torch::Tensor &gradOutGradFeatures,
                                                       const torch::Tensor &data);

template <c10::DeviceType>
torch::Tensor dispatchSplatIntoGridBezier(const GridBatchImpl &batchHdl, const JaggedTensor &points,
                                          const torch::Tensor &pointsData);

template <c10::DeviceType>
std::vector<JaggedTensor> dispatchVoxelsAlongRays(const GridBatchImpl &batchHdl,
                                                  const JaggedTensor  &rayOrigins,
                                                  const JaggedTensor &rayDirections, int64_t maxVox,
                                                  float eps, bool returnIjk, bool cumulative);

template <c10::DeviceType>
JaggedTensor dispatchSegmentsAlongRays(const GridBatchImpl &batchHdl,
                                       const JaggedTensor  &rayOrigins,
                                       const JaggedTensor &rayDirections, int64_t maxSegments,
                                       const double eps, const bool ignoreMasked);

template <c10::DeviceType>
JaggedTensor dispatchActiveGridCoords(const GridBatchImpl &gridAccessor, bool ignoreDisabledVoxels);

template <c10::DeviceType>
torch::Tensor dispatchTransformPointsToGrid(const GridBatchImpl &batchHdl,
                                            const JaggedTensor &points, bool isPrimal);

template <c10::DeviceType>
torch::Tensor dispatchInvTransformPointsToGrid(const GridBatchImpl &batchHdl,
                                               const JaggedTensor &points, bool isPrimal);

template <c10::DeviceType>
torch::Tensor dispatchTransformPointsToGridBackward(const GridBatchImpl &batchHdl,
                                                    const JaggedTensor &gradOut, bool isPrimal);

template <c10::DeviceType>
torch::Tensor dispatchInvTransformPointsToGridBackward(const GridBatchImpl &batchHdl,
                                                       const JaggedTensor &gradOut, bool isPrimal);

template <c10::DeviceType>
void dispatchVolumeRender(const torch::Tensor sigmas, const torch::Tensor rgbs,
                          const torch::Tensor deltas, const torch::Tensor ts,
                          const torch::Tensor raysAcc, const float opacityThreshold,
                          torch::Tensor &outOpacity, torch::Tensor &outDepth, torch::Tensor &outRgb,
                          torch::Tensor &outWs, torch::Tensor &outTotalSamples);

template <c10::DeviceType>
std::vector<JaggedTensor> dispatchMarchingCubes(const GridBatchImpl &batchHdl,
                                                const torch::Tensor &sdf, double level);

template <c10::DeviceType>
void dispatchVolumeRenderBackward(const torch::Tensor dLdOpacity, const torch::Tensor dLdDepth,
                                  const torch::Tensor dLdRgb, const torch::Tensor dLdWs,
                                  const torch::Tensor sigmas, const torch::Tensor rgbs,
                                  const torch::Tensor ws, const torch::Tensor deltas,
                                  const torch::Tensor ts, const torch::Tensor raysAcc,
                                  const torch::Tensor opacity, const torch::Tensor depth,
                                  const torch::Tensor rgb, const float opacityThreshold,
                                  torch::Tensor &outDLdSigmas, torch::Tensor &outDLdRbgs);

template <c10::DeviceType>
JaggedTensor dispatchIJKForMesh(const JaggedTensor &meshVertices, const JaggedTensor &meshFaces,
                                const std::vector<VoxelCoordTransform> &transforms);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForGrid(const GridBatchImpl &batchHdl, const nanovdb::Coord &bmin,
                                      const nanovdb::Coord &bmax);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForGridWithoutBorder(const GridBatchImpl  &batchHdl,
                                                   const nanovdb::Coord &bmin,
                                                   const nanovdb::Coord &bmax);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForPoints(const JaggedTensor &points, const nanovdb::Coord &bmin,
                                        const nanovdb::Coord                   &bmax,
                                        const std::vector<VoxelCoordTransform> &transforms);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForCoords(const JaggedTensor &coords, const nanovdb::Coord &bmin,
                                        const nanovdb::Coord &bmax);

template <c10::DeviceType>
JaggedTensor
dispatchNearestNeighborIJKForPoints(const JaggedTensor                     &points,
                                    const std::vector<VoxelCoordTransform> &transforms);

template <c10::DeviceType>
JaggedTensor dispatchCoarseIJKForFineGrid(const GridBatchImpl &batchHdl,
                                          nanovdb::Coord       coarseningFactor);

template <c10::DeviceType>
JaggedTensor dispatchFineIJKForCoarseGrid(const GridBatchImpl                 &batchHdl,
                                          nanovdb::Coord                       upsamplingFactor,
                                          const torch::optional<JaggedTensor> &maybeMask);

template <c10::DeviceType>
JaggedTensor dispatchConvIJKForGrid(const GridBatchImpl &batchHdl, const nanovdb::Coord &kernelSize,
                                    const nanovdb::Coord &stride);

template <c10::DeviceType>
torch::Tensor
dispatchScaledDotProductAttention(const torch::Tensor &query, const torch::Tensor &key,
                                  const torch::Tensor &value, const torch::Tensor &qLengths,
                                  const torch::Tensor &kvLengths, bool training, float scale);

// template <c10::DeviceType>
// std::tuple<torch::Tensor, torch::Tensor> dispatchQuatScaleToCovarPerciForward(
//     const torch::Tensor &quats,  // [N, 4]
//     const torch::Tensor &scales, // [N, 3]
//     const bool compute_covar,
//     const bool compute_preci,
//     const bool triu
// );

template <c10::DeviceType>
torch::Tensor dispatchSphericalHarmonicsForward(const int            sh_degree_to_use,
                                                const torch::Tensor &dirs,      // [N, 3]
                                                const torch::Tensor &sh_coeffs, // [N, ...]
                                                const torch::Tensor &radii      // [N]
);

template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward(const int            sh_degree_to_use,
                                   const torch::Tensor &dirs,      // [N, 3]
                                   const torch::Tensor &sh_coeffs, // [N, K, 3]
                                   const torch::Tensor &v_colors,
                                   const torch::Tensor &radii,     // [N]
                                   const bool           compute_v_dirs);

template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForward(const torch::Tensor &means,    // [N, 3]
                                  const torch::Tensor &quats,    // [N, 4]
                                  const torch::Tensor &scales,   // [N, 3]
                                  const torch::Tensor &viewmats, // [C, 4, 4]
                                  const torch::Tensor &Ks,       // [C, 3, 3]
                                  const uint32_t image_width, const uint32_t image_height,
                                  const float eps2d, const float near_plane, const float far_plane,
                                  const float radius_clip, const bool calc_compensations,
                                  const bool ortho);

template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionBackward(
    // fwd inputs
    const torch::Tensor               &means,         // [N, 3]
    const torch::Tensor               &quats,         // [N, 4]
    const torch::Tensor               &scales,        // [N, 3]
    const torch::Tensor               &viewmats,      // [C, 4, 4]
    const torch::Tensor               &Ks,            // [C, 3, 3]
    const at::optional<torch::Tensor> &compensations, // [N, 6] optional
    const uint32_t image_width, const uint32_t image_height, const float eps2d,
    // fwd outputs
    const torch::Tensor &radii,  // [C, N]
    const torch::Tensor &conics, // [C, N, 3]
    // grad outputs
    const torch::Tensor               &v_means2d,       // [C, N, 2]
    const torch::Tensor               &v_depths,        // [C, N]
    const torch::Tensor               &v_conics,        // [C, N, 3]
    const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
    const bool viewmats_requires_grad, const bool ortho);

template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersection(const torch::Tensor               &means2d, // [C, N, 2] or [M, 2]
                                 const torch::Tensor               &radii,   // [C, N] or [M]
                                 const torch::Tensor               &depths,  // [C, N] or [M]
                                 const at::optional<torch::Tensor> &camera_ids, // NULL or [M]
                                 const uint32_t num_cameras, const uint32_t tile_size,
                                 const uint32_t num_tiles_h, const uint32_t num_tiles_w);

template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dispatchGaussianRasterizeForward(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, D]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t image_origin_w,
    const uint32_t image_origin_h, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
);

template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeBackward(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, 3]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t image_origin_w,
    const uint32_t image_origin_h, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    const bool absgrad, const int64_t numSharedChannelsOverride = -1);

template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedForward(const torch::Tensor &g_sizes,  // [B] gaussian sizes
                                        const torch::Tensor &means,    // [ggz, 3]
                                        const torch::Tensor &quats,    // [ggz, 4] optional
                                        const torch::Tensor &scales,   // [ggz, 3] optional
                                        const torch::Tensor &c_sizes,  // [B] camera sizes
                                        const torch::Tensor &viewmats, // [ccz, 4, 4]
                                        const torch::Tensor &Ks,       // [ccz, 3, 3]
                                        const uint32_t image_width, const uint32_t image_height,
                                        const float eps2d, const float near_plane,
                                        const float far_plane, const float radius_clip,
                                        const bool ortho);

template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedBackward(const torch::Tensor &g_sizes,   // [B] gaussian sizes
                                         const torch::Tensor &means,     // [ggz, 3]
                                         const torch::Tensor &quats,     // [ggz, 4] optional
                                         const torch::Tensor &scales,    // [ggz, 3] optional
                                         const torch::Tensor &c_sizes,   // [B] camera sizes
                                         const torch::Tensor &viewmats,  // [ccz, 4, 4]
                                         const torch::Tensor &Ks,        // [ccz, 3, 3]
                                         const uint32_t image_width, const uint32_t image_height,
                                         const float          eps2d,
                                         const torch::Tensor &radii,     // [nnz]
                                         const torch::Tensor &conics,    // [nnz, 3]
                                         const torch::Tensor &v_means2d, // [nnz, 2]
                                         const torch::Tensor &v_depths,  // [nnz]
                                         const torch::Tensor &v_conics,  // [nnz, 3]
                                         const bool viewmats_requires_grad, const bool ortho);

template <c10::DeviceType>
fvdb::JaggedTensor
dispatchGaussianNanInfMask(const fvdb::JaggedTensor &means, const fvdb::JaggedTensor &quats,
                           const fvdb::JaggedTensor &scales, const fvdb::JaggedTensor &opacities,
                           const fvdb::JaggedTensor &sh_coeffs);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_OPS_H
