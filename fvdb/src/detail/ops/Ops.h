#pragma once

#include <torch/extension.h>

#include "detail/utils/Utils.h"
#include "detail/GridBatchImpl.h"
#include "Types.h"


namespace fvdb {
namespace detail {
namespace ops {

template <c10::DeviceType>
torch::Tensor dispatchJIdxForJOffsets(torch::Tensor joffsets, int64_t numElements);

template <c10::DeviceType>
JaggedTensor dispatchEnabledMask(const GridBatchImpl& batchHdl, bool returnDisabled);


template <c10::DeviceType>
torch::Tensor dispatchJIdxForGrid(const GridBatchImpl& batchHdl, bool ignoreDisabledVoxels);


template <c10::DeviceType>
nanovdb::GridHandle<PytorchDeviceBuffer> dispatchCreateNanoGridFromIJK(const JaggedTensor& ijk,
                                                                       bool isMutable);

template <c10::DeviceType>
nanovdb::GridHandle<PytorchDeviceBuffer> dispatchCreateNanoGridFromDense(uint32_t batchSize,
                                                                         nanovdb::Coord origin,
                                                                         nanovdb::Coord size,
                                                                         bool isMutable,
                                                                         torch::Device device,
                                                                         const torch::optional<torch::Tensor>& maybeMask);
template <c10::DeviceType>
void dispatchPopulateGridMetadata(const nanovdb::GridHandle<PytorchDeviceBuffer>& batchHdl,
                                  const std::vector<nanovdb::Vec3d>& voxelSizes,
                                  const std::vector<nanovdb::Vec3d>& voxelOrigins,
                                  const bool isMutable,
                                  torch::Tensor& outBatchOffsets,
                                  GridBatchImpl::GridMetadata* outPerGridMetadataHost,
                                  GridBatchImpl::GridMetadata* outPerGridMetadataDevice,
                                  GridBatchImpl::GridBatchMetadata* outBatchMetadataHost,
                                  GridBatchImpl::GridBatchMetadata* outBatchMetadataDevice) ;


template <c10::DeviceType>
void dispatchReadIntoDense(const GridBatchImpl& batchHdl,
                           const torch::Tensor& inGridData,
                           const torch::Tensor& denseOrigins,
                           torch::Tensor& outDenseTensor,
                           bool ignoreMasked);


template <c10::DeviceType>
void dispatchReadFromDense(const GridBatchImpl& batchHdl,
                           const torch::Tensor& inDenseTensor,
                           const torch::Tensor& denseOrigins,
                           torch::Tensor& outSparseTensor,
                           bool ignoreMasked);

template <c10::DeviceType>
void dispatchFillToGrid(const GridBatchImpl& fromGrid,
                        const GridBatchImpl& toGrid,
                        const torch::Tensor& fromFeatures,
                        torch::Tensor& toFeatures);

template <c10::DeviceType>
JaggedTensor dispatchIjkToInvIndex(const GridBatchImpl& batchHdl, const JaggedTensor& ijk);


template <c10::DeviceType>
JaggedTensor dispatchRayImplicitIntersection(const GridBatchImpl& batchHdl,
                                             const JaggedTensor& rayOrigins,
                                             const JaggedTensor& rayDirections,
                                             const JaggedTensor& gridScalars,
                                             float eps);


template <c10::DeviceType>
JaggedTensor dispatchCoordsInGrid(const GridBatchImpl& batchHdl,
                                  const JaggedTensor& coords, bool ignoreMasked);


template <c10::DeviceType>
int64_t dispatchCountEnabledVoxels(const GridBatchImpl& batchHdl, int batchIdx = -1);

template <c10::DeviceType>
JaggedTensor dispatchActiveVoxelsInBoundsMask(const GridBatchImpl& batchHdl,
                                              const Vec3iBatch& ijkMin,
                                              const Vec3iBatch& ijkMax,
                                              bool ignoreDisabledVoxels);

template <c10::DeviceType>
void dispatchSetMaskedIjk(const GridBatchImpl& batchHdl,
                          const JaggedTensor& coords,
                          bool maskedState);


template <c10::DeviceType>
std::vector<JaggedTensor> dispatchGridEdgeNetwork(const GridBatchImpl& gridHdl, bool returnVoxelCoordinates);


template <c10::DeviceType>
std::vector<JaggedTensor> dispatchUniformRaySamples(const GridBatchImpl& batchHdl,
                                                    const JaggedTensor& rayO,
                                                    const JaggedTensor& rayD,
                                                    const JaggedTensor& tMin,
                                                    const JaggedTensor& tMax,
                                                    const double minStepSize,
                                                    const double coneAngle,
                                                    const bool includeEndSegments);


template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridMaxPoolBackward(const GridBatchImpl& coarseBatchHdl,
                                                    const GridBatchImpl& fineBatchHdl,
                                                    const torch::Tensor& fineData,
                                                    const torch::Tensor& coarseGradOut,
                                                    nanovdb::Coord poolingFactor,
                                                    nanovdb::Coord stride);


template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridMaxPool(const GridBatchImpl& fineBatchHdl,
                                            const GridBatchImpl& coarseBatchHdl,
                                            const torch::Tensor& fineData,
                                            nanovdb::Coord poolingFactor,
                                            nanovdb::Coord stride);


template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridAvgPoolBackward(const GridBatchImpl& coarseBatchHdl,
                                                    const GridBatchImpl& fineBatchHdl,
                                                    const torch::Tensor& fineData,
                                                    const torch::Tensor& coarseGradOut,
                                                    nanovdb::Coord poolingFactor,
                                                    nanovdb::Coord stride);


template <c10::DeviceType>
torch::Tensor dispatchDownsampleGridAvgPool(const GridBatchImpl& fineBatchHdl,
                                            const GridBatchImpl& coarseBatchHdl,
                                            const torch::Tensor& fineData,
                                            nanovdb::Coord poolingFactor,
                                            nanovdb::Coord stride);


template <c10::DeviceType>
torch::Tensor dispatchUpsampleGridNearest(const GridBatchImpl& coarseBatchHdl,
                                          const GridBatchImpl& fineBatchHdl,
                                          const torch::Tensor& coarseData,
                                          nanovdb::Coord upsamplingFactor);


template <c10::DeviceType>
torch::Tensor dispatchUpsampleGridNearestBackward(const GridBatchImpl& fineBatchHdl,
                                                  const GridBatchImpl& coarseBatchHdl,
                                                  const torch::Tensor& gradOut,
                                                  const torch::Tensor& coarseData,
                                                  nanovdb::Coord upsamplingFactor);


template <c10::DeviceType>
JaggedTensor dispatchVoxelNeighborhood(const GridBatchImpl& batchHdl,
                                       const JaggedTensor& coords,
                                       nanovdb::Coord extentMin,
                                       nanovdb::Coord extentMax,
                                       int32_t shift);


template <c10::DeviceType>
JaggedTensor dispatchIjkToIndex(const GridBatchImpl& batchHdl, const JaggedTensor& ijk);


template <c10::DeviceType>
JaggedTensor dispatchPointsInGrid(const GridBatchImpl& batchHdl,
                                  const JaggedTensor& points,
                                  bool ignoreMasked);


template <c10::DeviceType>
JaggedTensor dispatchCubesInGrid(const GridBatchImpl& batchHdl,
                                 const JaggedTensor& cubeCenters,
                                 const Vec3dOrScalar& padMin,
                                 const Vec3dOrScalar& padMax,
                                 bool ignoreDisabledVoxels);

template <c10::DeviceType>
JaggedTensor dispatchCubesIntersectGrid(const GridBatchImpl& batchHdl,
                                        const JaggedTensor& cubeCenters,
                                        const Vec3dOrScalar& padMin,
                                        const Vec3dOrScalar& padMax,
                                        bool ignoreDisabledVoxels);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridTrilinear(const GridBatchImpl& batchHdl,
                                                       const JaggedTensor& points,
                                                       const torch::Tensor& gridData);


template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridTrilinearWithGrad(const GridBatchImpl& batchHdl,
                                                               const JaggedTensor& points,
                                                               const torch::Tensor& gridData);


template <c10::DeviceType>
torch::Tensor dispatchSampleGridTrilinearWithGradBackward(const GridBatchImpl& batchHdl,
                                                          const JaggedTensor& points,
                                                          const torch::Tensor& data,
                                                          const torch::Tensor& gradOutFeatures,
                                                          const torch::Tensor& gradOutGradFeatures);


template <c10::DeviceType>
torch::Tensor dispatchSplatIntoGridTrilinear(const GridBatchImpl& batchHdl,
                                             const JaggedTensor& points,
                                             const torch::Tensor& gridData);


template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridBezier(const GridBatchImpl& batchHdl,
                                                    const JaggedTensor& points,
                                                    const torch::Tensor& gridData);


template <c10::DeviceType>
std::vector<torch::Tensor> dispatchSampleGridBezierWithGrad(const GridBatchImpl& batchHdl,
                                                            const JaggedTensor& points,
                                                            const torch::Tensor& gridData);


template <c10::DeviceType>
torch::Tensor dispatchSampleGridBezierWithGradBackward(const GridBatchImpl& batchHdl,
                                                       const JaggedTensor& points,
                                                       const torch::Tensor& gradOutFeatures,
                                                       const torch::Tensor& gradOutGradFeatures,
                                                       const torch::Tensor& data);


template <c10::DeviceType>
torch::Tensor dispatchSplatIntoGridBezier(const GridBatchImpl& batchHdl,
                                          const JaggedTensor& points,
                                          const torch::Tensor& pointsData);


template <c10::DeviceType>
std::vector<JaggedTensor> dispatchVoxelsAlongRays(const GridBatchImpl& batchHdl,
                                                  const JaggedTensor& rayOrigins,
                                                  const JaggedTensor& rayDirections,
                                                  int64_t maxVox,
                                                  float eps);




template <c10::DeviceType>
std::vector<JaggedTensor> dispatchSegmentsAlongRays(const GridBatchImpl& batchHdl,
                                                          const JaggedTensor& rayOrigins,
                                                          const JaggedTensor& rayDirections,
                                                          int64_t maxSegments,
                                                          const double eps,
                                                          const bool ignoreMasked);


template <c10::DeviceType>
JaggedTensor dispatchActiveGridCoords(const GridBatchImpl& gridAccessor, bool ignoreDisabledVoxels);


template <c10::DeviceType>
torch::Tensor dispatchTransformPointsToGrid(const GridBatchImpl& batchHdl,
                                            const JaggedTensor& points,
                                            bool isPrimal);


template <c10::DeviceType>
torch::Tensor dispatchInvTransformPointsToGrid(const GridBatchImpl& batchHdl,
                                               const JaggedTensor& points,
                                               bool isPrimal);


template <c10::DeviceType>
torch::Tensor dispatchTransformPointsToGridBackward(const GridBatchImpl& batchHdl,
                                                    const JaggedTensor& gradOut,
                                                    bool isPrimal);


template <c10::DeviceType>
torch::Tensor dispatchInvTransformPointsToGridBackward(const GridBatchImpl& batchHdl,
                                                       const JaggedTensor& gradOut,
                                                       bool isPrimal);


template <c10::DeviceType>
void dispatchVolumeRender(const torch::Tensor sigmas,
                          const torch::Tensor rgbs,
                          const torch::Tensor deltas,
                          const torch::Tensor ts,
                          const torch::Tensor raysAcc,
                          const float opacityThreshold,
                          torch::Tensor& outOpacity,
                          torch::Tensor& outDepth,
                          torch::Tensor& outRgb,
                          torch::Tensor& outWs,
                          torch::Tensor& outTotalSamples);

template <c10::DeviceType>
std::vector<JaggedTensor> dispatchMarchingCubes(const GridBatchImpl& batchHdl,
                                                const torch::Tensor& sdf,
                                                double level);

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
                                  torch::Tensor& outDLdSigmas,
                                  torch::Tensor& outDLdRbgs);


template <c10::DeviceType>
JaggedTensor dispatchIJKForMesh(const JaggedTensor& meshVertices,
                                const JaggedTensor& meshFaces,
                                const std::vector<VoxelCoordTransform>& transforms);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForGrid(const GridBatchImpl& batchHdl,
                                      const nanovdb::Coord& bmin,
                                      const nanovdb::Coord& bmax);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForGridWithoutBorder(const GridBatchImpl& batchHdl,
                                                   const nanovdb::Coord& bmin,
                                                   const nanovdb::Coord& bmax);

template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForPoints(const JaggedTensor& points,
                                        const nanovdb::Coord& bmin,
                                        const nanovdb::Coord& bmax,
                                        const std::vector<VoxelCoordTransform>& transforms);


template <c10::DeviceType>
JaggedTensor dispatchPaddedIJKForCoords(const JaggedTensor& coords,
                                        const nanovdb::Coord& bmin,
                                        const nanovdb::Coord& bmax);


template <c10::DeviceType>
JaggedTensor dispatchNearestNeighborIJKForPoints(const JaggedTensor& points,
                                                 const std::vector<VoxelCoordTransform>& transforms);


template <c10::DeviceType>
JaggedTensor dispatchCoarseIJKForFineGrid(const GridBatchImpl& batchHdl,
                                          nanovdb::Coord coarseningFactor);


template <c10::DeviceType>
JaggedTensor dispatchFineIJKForCoarseGrid(const GridBatchImpl& batchHdl, nanovdb::Coord upsamplingFactor,
                                          const torch::optional<JaggedTensor>& maybeMask);

template <c10::DeviceType>
JaggedTensor dispatchConvIJKForGrid(const GridBatchImpl& batchHdl,
                                    const nanovdb::Coord& kernelSize, const nanovdb::Coord& stride);

template <c10::DeviceType>
torch::Tensor dispatchScaledDotProductAttention(const torch::Tensor& query, const torch::Tensor& key,
                                                const torch::Tensor& value,
                                                const torch::Tensor& qLengths, const torch::Tensor& kvLengths,
                                                bool training, float scale);

} // namespace ops
} // namespace detail
} // namespace fvdb
