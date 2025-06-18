// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_GRIDBATCH_H
#define FVDB_GRIDBATCH_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/Types.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>

#include <ATen/core/ivalue.h>

namespace fvdb {

struct GridBatch : torch::CustomClassHolder {
    // Set some speed limits so you don't shoot yourself in the foot
    constexpr static int64_t MAX_GRIDS_PER_BATCH = detail::GridBatchImpl::MAX_GRIDS_PER_BATCH;

    explicit GridBatch();
    explicit GridBatch(const torch::Device &device);
    GridBatch(nanovdb::GridHandle<detail::TorchDeviceBuffer> &&gridHdl,
              const std::vector<nanovdb::Vec3d> &voxelSizes,
              const std::vector<nanovdb::Vec3d> &voxelOrigins);

    /// @brief Return true if this is a contiguous view of the grid batch
    /// @return true if this is a contiguous view of the grid batch
    bool is_contiguous() const;

    /// @brief Return a contiguous copy of this grid batch. If the grid batch is already contiguous,
    ///        then return a reference to this
    /// @return A contiguous copy of this grid batch
    GridBatch contiguous() const;

    /// @brief Check if two GridBatches refer to the same underlying NanoVDB grid
    /// @param other Another GridBatch to compare with
    /// @return true if the two GridBatches refer to the same underlying NanoVDB grid
    bool is_same(const GridBatch &other) const;

    /// @brief Get the voxel size of the bi^th grid in the batch and return is a tensor of type
    /// dtype
    /// @param bi The batch index of the grid for which to get the voxel size
    /// @param dtype The dtype of the returned tensor
    /// @return A tensor of shape [3,] containing the voxel size of the bi^th grid in the batch
    torch::Tensor voxel_size_at(int64_t bi, const torch::Dtype &dtype = torch::kFloat32) const;

    /// @brief Get the voxel origin of the bi^th grid in the batch and return is a tensor of type
    /// dtype
    /// @param bi The batch index of the grid for which to get the voxel origin
    /// @param dtype The dtype of the returned tensor
    /// @return A tensor of shape [3,] containing the voxel origin of the bi^th grid in the batch
    torch::Tensor origin_at(int64_t bi, const torch::Dtype &dtype = torch::kFloat32) const;

    /// @brief Get the voxel size of all grids in this batch and return is a tensor of type dtype
    /// @param dtype The dtype of the returned tensor
    /// @return A tensor of shape [grid_count(), 3] containing the voxel size of all grids indexed
    /// by this batch
    torch::Tensor voxel_sizes(const torch::Dtype &dtype = torch::kFloat32) const;

    /// @brief Get the voxel origins of all grids in this batch and return is a tensor of type dtype
    /// @param dtype The dtype of the returned tensor
    /// @return A tensor of shape [grid_count(), 3] containing the voxel origins of all grids
    /// indexed by this batch
    torch::Tensor origins(const torch::Dtype &dtype = torch::kFloat32) const;

    /// @brief Get the number of grids indexed by this batch
    /// @return The number of grids indexed by this batch
    int64_t grid_count() const;

    /// @brief Get the total number of voxels indexed by this batch of grids
    /// @return The total number of voxels indexed by this batch of grids
    int64_t total_voxels() const;

    /// @brief Get the number of voxels indexed by the bi^th grid in the batch
    /// @param bi The batch index of the grid for which to get the number of voxels
    /// @return The number of voxels indexed by the bi^th grid in the batch
    int64_t num_voxels_at(int64_t bi) const;

    /// @brief Get the cumulative number of voxels indexed by the first bi+1 grids
    /// @param bi The batch index
    /// @return The cumulative number of voxels indexed by the first bi+1 grids
    int64_t cum_voxels_at(int64_t bi) const;

    /// @brief Get the number of voxels per grid indexed by this batch of grids
    /// @return An integer tensor containing the number of voxels per grid indexed by this batch
    torch::Tensor num_voxels() const;

    /// @brief Get the cumulative number of voxels indexed by the grids in this batch
    ///        i.e. [nvox_0, nvox_0+nvox_1, nvox_0+nvox_1+nvox_2, ...]
    /// @return An integer tensor containing the cumulative number of voxels indexed by the grids in
    /// this batch
    torch::Tensor cum_voxels() const;

    /// @brief Get the total number of bytes required to store all grids indexed by this batch
    /// @return The total number of bytes required to store all grids indexed by this batch
    int64_t total_bytes() const;

    /// @brief Get the number of bytes required to store each grid
    /// @return An integer tensor containing the number of bytes required to store each grid
    torch::Tensor num_bytes() const;

    /// @brief Get the total number of leaf nodes indexed by this batch of grids
    /// @return The total number of leaf nodes indexed by this batch of grids
    int64_t total_leaf_nodes() const;

    /// @brief Get the number of leaf nodes in each grid
    /// @return An integer tensor containing the number of leaf nodes in each grid
    torch::Tensor num_leaf_nodes() const;

    /// @brief Get the offsets of the voxels indexed by this batch of grid
    /// @return A tensor of shape [batch_size, 2] where the [bi, 0]^th entry is the offset of the
    /// first voxel
    ///         and the [bi, 1]^th entry is the offset one past the last voxel indexed by the bi^th
    ///         grid in the batch
    torch::Tensor joffsets() const;

    /// @brief Get the list indices for theis batch of grids
    /// @return A tensor of shape [total_grids, ldim] where the [i]^th entry is the list index of
    /// the i^th grid
    torch::Tensor jlidx() const;

    /// @brief Get the batch index for each voxel indexed by this batch of grids
    /// @return An integer tensor of shape [total_voxels,] where the [i]^th entry is the batch index
    /// of the i^th voxel
    torch::Tensor jidx() const;

    /// @brief Set the voxel size of all grids indexed by this batch to the specified value
    /// @param voxel_size A 3D (shape [3,]) tensor specifying the voxel size to set for each grid
    void set_global_voxel_size(const Vec3dOrScalar &voxel_size);

    /// @brief Set the voxel origin of all grids indexed by this batch to the specified value
    /// @param origin A 3D (shape [3,]) tensor specifying the voxel origin to set for each grid
    void set_global_origin(const Vec3d &origin);

    /// @brief Get the device on which this grid is stored
    /// @return The device on which this grid is stored
    c10::Device device() const;

    /// @brief Get the primal transforms of the grids in this batch (i.e. world to primal grid
    /// coordinates)
    /// @return A std::vector<VoxelCoordTransform> containing the primal transforms of the grids in
    /// this batch
    const std::vector<detail::VoxelCoordTransform> primal_transforms() const;

    /// @brief Get the dual transforms of the grids in this batch (i.e. world to dual grid
    /// coordinates)
    /// @return A std::vector<detail::VoxelCoordTransform> containing the dual transforms of the
    /// grids in this batch
    const std::vector<detail::VoxelCoordTransform> dual_transforms() const;

    /// @brief Get the primal transform of the bi^th grid in the batch (i.e. world to primal grid
    /// coordinates)
    /// @param bi The index of the grid in the batch for which to get the primal transform
    /// @return The primal transform of the bi^th grid in the batch
    const fvdb::detail::VoxelCoordTransform primal_transform_at(int64_t bi) const;

    /// @brief Get the dual transform of the bi^th grid in the batch (i.e. world to dual grid
    /// coordinates)
    /// @param bi The index of the grid in the batch for which to get the dual transform
    /// @return The dual transform of the bi^th grid in the batch
    const fvdb::detail::VoxelCoordTransform dual_transform_at(int64_t bi) const;

    /// @brief Get the bounding box (in voxel coordinates) for each grid in the batch
    /// @return A tensor bboxes of shape [B, 2, 3] where
    ///         bboxes[bi] = [[bmin_i, bmin_j, bmin_z=k], [bmax_i, bmax_j, bmax_k]] is the bi^th
    ///         bounding box such that bmin <= ijk < bmax for all voxels ijk in the bi^th grid
    const torch::Tensor bbox() const;

    /// @brief Get the bounding box (in voxel coordinates) of the bi^th grid in the batch
    /// @return A tensor, bbox, of shape [2, 3] where
    ///         bbox = [[bmin_i, bmin_j, bmin_z=k], [bmax_i, bmax_j, bmax_k]] is the bi^th bounding
    ///         box such that bmin <= ijk < bmax for all voxels ijk in the bi^th grid
    const torch::Tensor bbox_at(int64_t bi) const;

    /// @brief Get the bounding box (in voxel coordinates) for the dual of each grid in the batch
    /// @return A tensor bboxes of shape [B, 2, 3] where
    ///         bboxes[bi] = [[bmin_i, bmin_j, bmin_z=k], [bmax_i, bmax_j, bmax_k]] is the bi^th
    ///         bounding box such that bmin <= ijk < bmax for all voxels ijk in the dual of the
    ///         bi^th grid
    const torch::Tensor dual_bbox() const;

    /// @brief Get the bounding box (in voxel coordinates) of the dual of the bi^th grid in the
    /// batch
    /// @return A tensor, bbox, of shape [2, 3] where
    ///         bbox = [[bmin_i, bmin_j, bmin_z=k], [bmax_i, bmax_j, bmax_k]] is the bi^th bounding
    ///         box such that bmin <= ijk < bmax for all voxels ijk in the dual of the bi^th grid
    const torch::Tensor dual_bbox_at(int64_t bi) const;

    /// @brief Get the bounding box (in voxel coordinates) which contains all the grids in this
    /// batch
    /// @return A tensor, total_bbox, of shape [2, 3] where
    ///         total_bbox = [[bmin_i, bmin_j, bmin_z=k], [bmax_i, bmax_j, bmax_k]] is the bounding
    ///         box such that bmin <= ijk < bmax for all voxels ijk in the batch
    const torch::Tensor total_bbox() const;

    /// @brief Downsample this batch of grids using maxpooling
    /// @param pool_factor How much to pool by (i,e, (2,2,2) means take max over 2x2x2 from start of
    /// window)
    /// @param data Data at each voxel in this grid to be downsampled (JaggedTensor of shape [B, -1,
    /// *])
    /// @param stride The stride to use when pooling
    /// @param coarse_grid An optional coarse grid used to specify the output. This is mainly used
    /// for memory
    ///                    efficiency so you can chache grids. If you don't pass it in, we'll just
    ///                    create it for you.
    /// @return A pair (coarseData, coarseGrid) where coarseData is a JaggedTensor of shape [B, -1,
    /// *] of downsampled data
    ///         and coarseGrid is a GridBatch representing the downsampled grid batch
    std::pair<JaggedTensor, GridBatch>
    max_pool(Vec3iOrScalar pool_factor,
             const JaggedTensor &data,
             Vec3iOrScalar stride                 = 0,
             std::optional<GridBatch> coarse_grid = std::nullopt) const;

    /// @brief Downsample this batch of grids using average pooling
    /// @param pool_factor How much to pool by (i,e, (2, 2, 2) means take max over 2x2x2 from start
    /// of window)
    /// @param data Data at each voxel in this grid to be downsampled (JaggedTensor of shape [B, -1,
    /// *])
    /// @param stride The stride to use when pooling
    /// @param coarse_grid An optional coarse grid used to specify the output. This is mainly used
    /// for memory
    ///                    efficiency so you can chache grids. If you don't pass it in, we'll just
    ///                    create it for you.
    /// @return A pair (coarseData, coarseGrid) where coarseData is a JaggedTensor of shape [B, -1,
    /// *] of downsampled data
    ///         and coarseGrid is a GridBatch representing the downsampled grid batch
    std::pair<JaggedTensor, GridBatch>
    avg_pool(Vec3iOrScalar pool_factor,
             const JaggedTensor &data,
             Vec3iOrScalar stride                 = 0,
             std::optional<GridBatch> coarse_grid = std::nullopt) const;

    /// @brief Subdivide this batch of grids using nearest neighbor interpolation
    /// @param subdiv_factor How much to upsample by (i,e, (2,2,2) means upsample by 2x2x2)
    /// @param data Data at each voxel in this grid to be upsampled (JaggedTensor of shape [B, -1,
    /// *])
    /// @param mask An optional mask of shape [B, -1] specifying which coarse voxels to upsample
    /// @param fine_grid An optional coarse grid used to specify the output. This is mainly used for
    /// memory
    ///                  efficiency so you can chache grids. If you don't pass it in, we'll just
    ///                  create it for you.
    /// @return A pair (fineData, fineGrid) where fineData is a JaggedTensor of shape [B, -1, *] of
    /// upsampled data and
    ///         fineGrid is a GridBatch representing the upsampled grid batch
    std::pair<JaggedTensor, GridBatch>
    subdivide(Vec3iOrScalar subdiv_factor,
              const JaggedTensor &data,
              const std::optional<JaggedTensor> mask = std::nullopt,
              std::optional<GridBatch> fine_grid     = std::nullopt) const;

    /// @brief Read the values from a dense tensor of the voxels at the specified coordinates
    /// @param dense_data A dense tensor of shape [B, W, H, D, *]
    /// @param dense_origins A tensor of shape [B, 3] or [3,] specifying the voxel coordinate(s) of
    /// the origin of the dense tensor i.e. [:, 0, 0, 0]
    /// @return A JaggedTensor with shape [B, -1, *] containing the values at the specified
    /// coordinates
    JaggedTensor
    read_from_dense(const torch::Tensor &dense_data,
                    const Vec3iBatch &dense_origins = torch::zeros(3, torch::kInt32)) const;

    /// @brief Read the values from a JaggedTensor indexed by this batch into a dense tensor
    /// @param sparse_data A JaggedTensor of shape [B, -1, *] containing one value per voxel in the
    /// batch
    /// @param min_coord An optional minimum coordinate to read from the batch (in voxel
    /// coordinates).
    ///                  Defaults to the minimum coordinate of the batch.
    /// @param grid_size An optional grid size to read from the batch (in voxel coordinates).
    ///                  Defaults to the total size of a grid containing the whole batch.
    /// @return A dense tensor of shape [B, W, H, D, *] containing the values at the specified
    /// coordinates (and zero elsewhere)
    torch::Tensor write_to_dense(const JaggedTensor &sparse_data,
                                 const std::optional<Vec3iBatch> &min_coord = std::nullopt,
                                 const std::optional<Vec3i> &grid_size      = std::nullopt) const;

    /// @brief Given a GridBatch and features associated with it,
    ///        return a JaggedTensor representing features for this batch of grids.
    ///        Fill any voxels not in the GridBatch with the default value.
    /// @param other_features A JaggedTensor of shape [B, -1, *] containing features associated with
    /// other_grid.
    /// @param other_grid A GridBatch representing the grid to fill from.
    /// @param default_value The value to fill in for voxels not in other_grid.
    JaggedTensor fill_from_grid(const JaggedTensor &other_features,
                                const GridBatch &other_grid,
                                float default_value = 0.0f) const;

    /// @brief Convert grid coordinates to world coordinates
    /// @param ijk A JaggedTensor of grid coordinates with shape [B, -1, 3] (one point set per grid
    /// in the batch)
    /// @return A JaggedTensor of world coordinates with shape [B, -1, 3] (one point set per grid in
    /// the batch)
    JaggedTensor grid_to_world(const JaggedTensor &ijk) const;

    /// @brief Convert world coordinates to grid coordinates
    /// @param points A JaggedTensor of world coordinates with shape [B, -1, 3] (one point set per
    /// grid in the batch)
    /// @return A JaggedTensor of grid coordinates with shape [B, -1, 3] (one point set per grid in
    /// the batch)
    JaggedTensor world_to_grid(const JaggedTensor &points) const;

    /// @brief Get grid-to-world matrices
    /// @return A JaggedTensor of grid-to-world matrices with shape [B, 4, 4]
    torch::Tensor grid_to_world_matrices(const torch::Dtype &dtype = torch::kFloat32) const;

    /// @brief Get world-to-grid matrices
    /// @return A JaggedTensor of world-to-grid matrices with shape [B, 4, 4]
    torch::Tensor world_to_grid_matrices(const torch::Dtype &dtype = torch::kFloat32) const;

    /// @brief Sample features on the grid batch using trilinear interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the
    /// batch)
    /// @param voxel_data a JaggedTensor of C-dimensional features at each voxel with shape [B, -1,
    /// C] or a Tensor of
    ///                   shape [N, C] where N is the total number of voxels in the batch
    ///                   (one item for each voxel in each grid in the batch)
    /// @return a JaggedTensor of sampled data with shape [B, -1, C] (one sample set per point)
    JaggedTensor sample_trilinear(const JaggedTensor &points, const JaggedTensor &voxel_data) const;

    /// @brief Sample features and spatial gradients on the grid batch using trilinear interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the
    /// batch)
    /// @param voxel_data a JaggedTensor of C-dimensional features at each voxel with shape [B, -1,
    /// C] or a Tensor of
    ///                   shape [N, C] where N is the total number of voxels in the batch
    ///                   (one item for each voxel in each grid in the batch)
    /// @return a pair (feat, grad_feat) which are JaggedTensors of sampled data with shape [B, -1,
    /// C], and [B, -1, C, 3]
    ///         respectively where feat are the sampled features and grad_feat are the spatial
    ///         gradients of the sampled features (one sample set per point)
    std::vector<JaggedTensor> sample_trilinear_with_grad(const JaggedTensor &points,
                                                         const JaggedTensor &voxel_data) const;

    /// @brief Sample features on the grid batch using bezier interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the
    /// batch)
    /// @param voxel_data a JaggedTensor of C-dimensional features at each voxel with shape [B, -1,
    /// C] or a Tensor of
    ///                   shape [N, C] where N is the total number of voxels in the batch
    ///                   (one item for each voxel in each grid in the batch)
    /// @return a JaggedTensor of sampled data with shape [B, -1, C] (one sample set per point)
    JaggedTensor sample_bezier(const JaggedTensor &points, const JaggedTensor &voxel_data) const;

    /// @brief Sample features and spatial gradients on the grid batch using bezier interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the
    /// batch)
    /// @param voxel_data a JaggedTensor of C-dimensional features at each voxel with shape [B, -1,
    /// C] or a Tensor of
    ///                   shape [N, C] where N is the total number of voxels in the batch
    ///                   (one item for each voxel in each grid in the batch)
    /// @return a pair (feat, grad_feat) which are JaggedTensors of sampled data with shape [B, -1,
    /// C], and [B, -1, C, 3]
    ///         respectively where feat are the sampled features and grad_feat are the spatial
    ///         gradients of the sampled features (one sample set per point)
    std::vector<JaggedTensor> sample_bezier_with_grad(const JaggedTensor &points,
                                                      const JaggedTensor &voxel_data) const;

    /// @brief Splat features at points into a grid batch using trilinear interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the
    /// batch)
    /// @param points_data a JaggedTensor of C-dimensional features at each point with shape [B, -1,
    /// C]
    /// @return a JaggedTensor of C-dimensional features at each voxel in the batch with shape [B,
    /// -1, C]
    JaggedTensor splat_trilinear(const JaggedTensor &points, const JaggedTensor &points_data) const;

    /// @brief Splat features at points into a grid using bezier interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the
    /// batch)
    /// @param points_data a JaggedTensor of C-dimensional features at each point with shape [B, -1,
    /// C]
    /// @return a JaggedTensor of C-dimensional features at each voxel in the batch with shape [B,
    /// -1, C]
    JaggedTensor splat_bezier(const JaggedTensor &points, const JaggedTensor &points_data) const;

    /// @brief Get the indices of neighbors in the N-ring of each voxel in the grid batch
    ///        (possibly bitshifting the coordinates which is useful when you use multiple grids to
    ///        represent different levels of a hierarchy and you want to query this grid with
    ///        coordinates at a finer level)
    /// @param ijk A JaggedTensor of voxel coordinates with shape [B, -1, 3] (one set of coordinates
    /// per grid in the batch)
    /// @param extent The size of a neighborhood to find indexes
    /// @param bitshift The number of bits to shift the coordinates by
    /// @return A JaggedTensor of neighbor indexes with shape [B, -1, 2*extent+1, 2*extent+1,
    /// 2*extent+1] (-1 value indicates no neighbor at that index)
    JaggedTensor
    neighbor_indexes(const JaggedTensor &ijk, int32_t extent, int32_t bitshift = 0) const;

    /// @brief Return whether each point lies inside the grid batch
    /// @param points A JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the
    /// batch)
    /// @return A JaggedTensor of booleans with shape [B, -1] (one boolean per point)
    ///         where the [bi, i]^th entry is true if points[bi, i] lies inside the bi^th grid in
    ///         the batch
    JaggedTensor points_in_active_voxel(const JaggedTensor &points) const;

    /// @brief Return whether the cube with corners at cube_min and cube_max centered at each point
    /// in world space intersect the grid batch
    /// @param cube_centers A JaggedTensor of points with shape [B, -1, 3] (one point set per grid
    /// in the batch)
    /// @param cube_min A 3D tensor specifying the min corner relative to each point to check
    /// @param cube_max A 3D tensor specifying the max corner relative to each point to check
    /// @return A JaggedTensor of booleans with shape [B, -1] (one boolean per point)
    ///         where the [bi, i]^th entry is true if the cube with extent (min, max) + points[bi,
    ///         i] intersects the bi^th grid in the batch
    JaggedTensor cubes_intersect_grid(const JaggedTensor &cube_centers,
                                      const Vec3dOrScalar &cube_min = 0.0,
                                      const Vec3dOrScalar &cube_max = 0.0) const;

    /// @brief Return whether the cube with corners at cube_min and cube_max centered at each point
    /// in world space is fully contained in the grid batch's stencil
    /// @param cube_centers A JaggedTensor of points with shape [B, -1, 3] (one point set per grid
    /// in the batch)
    /// @param cube_min A 3D tensor specifying the min corner relative to each point to check
    /// @param cube_max A 3D tensor specifying the max corner relative to each point to check
    /// @return A JaggedTensor of booleans with shape [B, -1] (one boolean per point)
    ///         where the [bi, i]^th entry is true if the cube with extent (min, max) + points[bi,
    ///         i] lies inside the bi^th grid in the batch
    JaggedTensor cubes_in_grid(const JaggedTensor &cube_centers,
                               const Vec3dOrScalar &cube_min = 0.0,
                               const Vec3dOrScalar &cube_max = 0.0) const;

    /// @brief Return whether each coordinate is in the grid batch or not
    /// @param ijk A JaggedTensor of ijk coordinates with lshape [N_0, ..., N_B] and eshape (3,)
    ///            (one coordinate set per grid in the batch)
    /// @return A JaggedTensor of booleans with shape [B, -1] (one boolean per coordinate)
    ///         where the [bi, i]^th entry is true if coords[bi, i] lies inside the bi^th grid in
    ///         the batch
    JaggedTensor coords_in_active_voxel(const JaggedTensor &ijk) const;

    /// @brief Return the integer offset of each ijk value in the grid batch
    /// @param ijk A JaggedTensor of ijk coordinates with shape [B, -1, 3] (one coordinate set per
    /// grid in the batch)
    /// @param cumulative Whether to return cumulative offsets in the batch or offsets relative to
    /// each grid
    /// @return A JaggedTensor of integer offsets with shape [B, -1] into the grid batch (one offset
    /// per coordinate)
    JaggedTensor ijk_to_index(const JaggedTensor &ijk, bool cumulative = false) const;

    /// @brief Return a JaggedTensor of integers such that if it is used as a permutation of the
    /// input IJK coordinates,
    ///        it will re-order them to the indexing order of the grid batch. This effectively
    ///        performs the inverse of ijk_to_index if you pass in the ijk coordinates in the grid.
    ///        i.e. output[ijk_to_index(ijk[i])] = i
    /// @param ijk A JaggedTensor of ijk coordinates with lshape [N_0, ..., N_B] and eshape (3,)
    ///            (one coordinate set per grid in the batch)
    /// @param cumulative Whether to return cumulative offsets in the batch or offsets relative to
    /// each grid
    /// @return A JaggedTensor of integers with shape [B, -1] (one integer per grids' ijk) which
    /// inverts ijkToIndex
    JaggedTensor ijk_to_inv_index(const JaggedTensor &ijk, bool cumulative = false) const;

    /// @brief Return the set of active ijk coordinates indexed by this grid batch
    /// @return A JaggedTensor of voxel coordinates indexed by this grid batch (shape [B, -1, 3])
    JaggedTensor ijk() const;

    /// @brief Find the intersection between a collection of rays and the zero level set of a scalar
    /// field
    ///        at each voxel in the grid batch
    /// @param ray_origins A JaggedTensor of ray origins with shape [B, -1, 3] (one ray set per grid
    /// in the batch)
    /// @param ray_directions A JaggedTensor of ray directions with shape [B, -1, 3] (one ray set
    /// per grid in the batch)
    /// @param grid_scalars A JaggedTensor of scalar values with shape [B, -1] (one scalar per voxel
    /// in the batch)
    /// @param eps Skip voxels where the ray intersects by less than this distance
    /// @return A JaggedTensor of intersection times with shape [B, -1] (one time per ray)
    JaggedTensor ray_implicit_intersection(const JaggedTensor &ray_origins,
                                           const JaggedTensor &ray_directions,
                                           const JaggedTensor &grid_scalars,
                                           double eps = 0.0) const;

    /// @brief Enumerate the voxels in this grid batch (in-sorted order) intersected by a collection
    /// of rays
    /// @param ray_origins A JaggedTensor of ray origins with lshape [N_0, ..., N_B] and eshape [3,]
    ///                    where N_i is the number of rays to intersect with the i^th grid
    /// @param ray_directions A JaggedTensor of ray directions with lshape [N_0, ..., N_B] and
    /// eshape [3,]
    ///                       where N_i is the number of rays to intersect with the i^th grid
    /// @param max_voxels The maximum number of voxels to return per ray
    /// @param eps Skip voxels where the ray intersects by less than this distance
    /// @param return_ijk Whether to return the voxel coordinates in the grid or world coordinates
    /// or the voxel index
    /// @param cumulative Whether to return cumulative indices in the batch or indices relative to
    /// each grid
    ///                   (only applicable to return_ijk = false, otherwise ignored)
    /// @return A pair of JaggedTensors containing the voxels (or voxel indices) intersected by the
    /// rays. i.e.:
    ///             - voxels: A JaggedTensor with lshape [[V_{0,0}, ..., V_{0,N_0}], ..., [V_{B,0},
    ///             ..., V_{B,N_B}]]
    ///                       and eshape (3,) or (,) containing the ijk coordinates or indices of
    ///                       the voxels
    ///             - times: A JaggedTensor with lshape [[T_{0,0}, ..., T_{0,N_0}], ..., [T_{B,0},
    ///             ..., T_{B,N_B}]]
    ///                      and eshape (2,) containg the entry and exit distance along the ray of
    ///                      each voxel
    std::vector<JaggedTensor> voxels_along_rays(const JaggedTensor &ray_origins,
                                                const JaggedTensor &ray_directions,
                                                int64_t max_voxels,
                                                double eps      = 0.0,
                                                bool return_ijk = true,
                                                bool cumulative = false) const;

    /// @brief Enumerate the continuous segments (regions which overlap active voxels) in this
    ///        grid batch (in-sorted order) intersected by a collection of rays
    /// @param ray_origins A JaggedTensor of ray origins with lshape [N_0, ..., N_B] and eshape [3,]
    ///                    where N_i is the number of rays to intersect with the i^th grid
    /// @param ray_directions A JaggedTensor of ray directions with lshape [N_0, ..., N_B] and
    /// eshape [3,]
    ///                       where N_i is the number of rays to intersect with the i^th grid
    /// @param max_segments The maximum number of segments to return per ray
    /// @param eps Skip segments whose length is less than this distance
    /// @return A JaggedTensor containing the segments intersected by the rays. i.e. a JaggedTensor
    ///         with lshape [[S_{0,0}, ..., S_{0,N_0}], ..., [S_{B,0}, ..., S_{B,N_B}]]
    JaggedTensor segments_along_rays(const JaggedTensor &ray_origins,
                                     const JaggedTensor &ray_directions,
                                     int64_t max_segments,
                                     double eps = 0.0) const;

    /// @brief Generate a set of uniform samples in active regions along a specified set of rays
    /// @param ray_origins A JaggedTensor of ray origins with lshape [N_0, ..., N_B] and eshape [3,]
    ///                    where N_i is the number of rays to intersect with the i^th grid
    /// @param ray_directions A JaggedTensor of ray directions with lshape [N_0, ..., N_B] and
    /// eshape [3,]
    ///                       where N_i is the number of rays to intersect with the i^th grid
    /// @param t_min The start distance along each ray to begin generating samples
    /// @param t_max The end distance along each ray to stop generating samples
    /// @param step_size The distance between samples along each ray
    /// @param cone_angle A cone angle for each ray used to space samples along the ray
    /// @param include_end_segments Whether to include the end segments of the rays in the samples
    /// @param return_midpoints Whether to return the midpoint of each sample instead of the start
    /// and end
    /// @param eps Skip segments whose length is less than this distance
    /// @return A JaggedTensor containing the samples along the rays. i.e. a JaggedTensor
    ///         with lshape [[S_{0,0}, ..., S_{0,N_0}], ..., [S_{B,0}, ..., S_{B,N_B}]] and eshape
    ///         (2,) or (1,) representing the start and end distance of each sample or the midpoint
    ///         of each sample if return_midpoints is true
    JaggedTensor uniform_ray_samples(const JaggedTensor &ray_origins,
                                     const JaggedTensor &ray_directions,
                                     const JaggedTensor &t_min,
                                     const JaggedTensor &t_max,
                                     double step_size,
                                     double cone_angle         = 0.0,
                                     bool include_end_segments = true,
                                     bool return_midpoints     = false,
                                     double eps                = 0.0) const;

    /// @brief Return an edge network used which can be used to plot the grids in this batch
    /// @param return_voxel_coordinates Whether to return the vertices in voxel coordinates or world
    /// coordinates
    /// @return A pair (verts, edges) where verts is a JaggedTensor of vertex positions with shape
    /// [B, -1, 3]
    ///         (one vertex set per grid in the batch) and edges is a JaggedTensor of edge indices
    ///         of shape [B, -1, 2] (one edge set per grid in the batch)
    std::vector<JaggedTensor> viz_edge_network(bool return_voxel_coordinates = false) const;

    /// @brief Return a batch of grids representing the dual of this batch. i.e. The centers of the
    /// dual grid correspond
    ///        to the corners of this grid batch. The [i, j, k] coordinate of the dual grid
    ///        corresponds to the bottom/left/back corner of the [i, j, k] voxel in this grid batch.
    /// @param exclude_border Whether to exclude the border of the grid batch when computing the
    /// dual grid
    /// @return A GridBatch representing the dual of this grid batch
    GridBatch dual_grid(bool exclude_border = false) const;

    /// @brief Return a batch of grids representing the coarsened version of this batch.
    ///        Each voxel [i, j, k] in this grid batch maps to voxel [i / branchFactor, j /
    ///        branchFactor, k / branchFactor] in the coarse batch.
    /// @param coarsening_factor The factor by which to coarsen the grid batch (i.e (2, 2, 2)
    /// coarses by a factor of 2x2x2)
    /// @return A GridBatch representing the coarsened version of this batch.
    GridBatch coarsened_grid(Vec3iOrScalar coarsening_factor) const;

    /// @brief Subdivide the grid batch into a finer grid batch.
    ///        Each voxel [i, j, k] in this grid batch maps to voxels [i * subdivFactor, j *
    ///        subdivFactor, k * subdivFactor] in the fine batch.
    /// @param subdiv_factor The factor by which to subdivide the grid batch
    /// @param mask An optional JaggedTensor of shape [B, -1] of boolean values indicating which
    /// voxels to subdivide
    /// @return A GridBatch representing the subdivided version of this batch.
    GridBatch subdivided_grid(Vec3iOrScalar subdiv_factor,
                              const std::optional<JaggedTensor> mask = std::nullopt) const;

    /// @brief Return a batch of grids representing the clipped version of this batch of grids.
    /// @param ijk_min Index space minimum bound of the clip region.
    /// @param ijk_max Index space maximum bound of the clip region.
    /// @return A GridBatch representing the clipped version of this batch of grids.
    GridBatch clipped_grid(const Vec3iBatch &ijk_min, const Vec3iBatch &ijk_max) const;

    /// @brief Generate the grid that is affected by the convolution operator.
    /// @param kernel_size The kernel size of convolution
    /// @param stride The stride of the convolution
    /// @return A GridBatch representing the convolved grid.
    GridBatch conv_grid(Vec3iOrScalar kernel_size, Vec3iOrScalar stride) const;

    /// @brief Return a batch of grids representing the dilated version of this batch of grids.
    /// @param dilation The dilation factor of the grid batch
    /// @return A GridBatch representing the dilated version of this batch of grids.
    GridBatch dilated_grid(const int dilation) const;

    /// @brief Return a batch of grids representing the merged version of this batch of grids with a
    /// second batch of grids
    /// @param other The second grid batch to merge into this one
    /// @return A GridBatch representing the union of both grid batches
    GridBatch merged_grid(const GridBatch &other) const;

    /// @brief Return a batch of grids representing the clipped version of this batch of grids and
    /// corresponding features.
    /// @param features A JaggedTensor of shape [B, -1, *] containing features associated with this
    /// batch of grids.
    /// @param ijk_min Index space minimum bound of the clip region.
    /// @param ijk_max Index space maximum bound of the clip region.
    /// @return A pair (clipped_features, clipped_grid) where clipped_features is a JaggedTensor of
    /// shape [B, -1, *] and
    ///         clipped_grid is a GridBatch representing the clipped version of this batch of grids.
    std::pair<JaggedTensor, GridBatch>
    clip(const JaggedTensor &features, const Vec3iBatch &ijk_min, const Vec3iBatch &ijk_max) const;

    /// @brief Extract 0-isosurface from an implicit field.
    /// @param field implicit value stored on each voxel center (or voxel corner on a dual grid)
    /// @param level level set of the surface to extract
    /// @return vertices and faces arrays of the extracted isosurface
    std::vector<JaggedTensor> marching_cubes(const JaggedTensor &field, double level = 0.0) const;

    /// @brief Perform in-grid convolution using fast halo buffer method. Currently only supports
    /// kernel_size = 3.
    /// @param features A JaggedTensor of shape [B, -1, *] containing features associated with this
    /// batch of grids.
    /// @param kernel A tensor of shape [Out, In, 3, 3, 3] containing the kernel to convolve with.
    /// @return A JaggedTensor of shape [B, -1, *] containing the convolved features.
    JaggedTensor
    sparse_conv_halo(const JaggedTensor &features, const torch::Tensor &kernel, int variant) const;

    /// @brief Return a grid batch on the specified device. If the passed in device is the same as
    /// this grid batch's
    ///        device, then this grid batch is returned. Otherwise, a copy of this grid batch is
    ///        returned on the specified device.
    /// @param to_device The device to return the grid batch on
    /// @return A GridBatch representing this grid batch on the specified device
    GridBatch to(const torch::Device &to_device) const;

    /// @brief Return a view of this grid batch containing the grid at the specified index i.e.
    /// grid_batch[bi]
    /// @param bi The index to get a view on
    /// @return A GridBatch representing the grid at the specified index
    GridBatch index(int64_t bi) const;

    /// @brief Return a slice view of this grid batch i.e. grid_batch[start:stop:step]
    /// @param start The start index of the slice
    /// @param stop The stop index of the slice
    /// @param step The step of the slice
    /// @return A GridBatch representing the slice of this grid batch
    GridBatch index(size_t start, size_t stop, size_t step) const;

    /// @brief Return a view of this grid batch at the specified indices i.e. grid_batch[[i1, i2,
    /// ...]]
    /// @param bi A list of integers representing the indices to get a view on
    /// @return The grid batch vieweed at the specified indices
    GridBatch index(const std::vector<int64_t> &bi) const;

    /// @brief Return a view of this grid batch at indices specified by the given mask i.e.
    /// grid_batch[mask]
    /// @param bi A list of integers representing the indices to get a view on
    /// @return The grid batch vieweed at the specified indices
    GridBatch index(const std::vector<bool> &bi) const;

    /// @brief Return a view of this grid batch at the specified indices (or mask if bi is a bool
    /// tensor) i.e. grid_batch[[i1, i2, ...]]
    /// @param bi A list of integers representing the indices to get a view on
    /// @return The grid batch vieweed at the specified indices
    GridBatch index(const torch::Tensor &bi) const;

    /// @brief Return a JaggedTensor whose joffsets and jidx match this grid batch's
    /// @param data The data to use for the JaggedTensor (first dimension must match the total
    /// number of voxels in the grid batch)
    /// @return A JaggedTensor corresponding to the voxel grid of this grid batch
    JaggedTensor jagged_like(const torch::Tensor &data) const;

    /// @brief Populate the grid batch with voxels that intersect a triangle mesh
    /// @param vertices A JaggedTensor of shape [B, -1, 3] containing one vertex set per grid to
    /// create
    /// @param faces  A JaggedTensor of shape [B, -1, 3] containing one face set per grid to create
    /// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid
    /// in the batch or one voxel size for all grids
    /// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the
    /// [0, 0, 0] voxel
    ///                for each grid in the batch, or one origin for all grids
    void set_from_mesh(const JaggedTensor &vertices,
                       const JaggedTensor &faces,
                       const Vec3dBatchOrScalar &voxel_sizes = 1.0,
                       const Vec3dBatch &origins             = torch::zeros(3, torch::kInt32));

    /// @brief Populate the grid batch with voxels which contain a point in an input set of point
    /// clouds
    /// @param points A JaggedTensor with shape [B, -1, 3] containing one point set per grid to
    /// create
    /// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid
    /// in the batch or one voxel size for all grids
    /// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the
    /// [0, 0, 0] voxel
    ///                for each grid in the batch, or one origin for all grids
    void set_from_points(const JaggedTensor &points,
                         const Vec3dBatchOrScalar &voxel_sizes = 1.0,
                         const Vec3dBatch &origins             = torch::zeros(3, torch::kInt32));

    /// @brief Populate the grid batch with the eight nearest voxels to each point in an input set
    /// of point clouds
    /// @param points A JaggedTensor with shape [B, -1, 3] containing one point set per grid to
    /// create
    /// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid
    /// in the batch or one voxel size for all grids
    /// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the
    /// [0, 0, 0] voxel
    ///                for each grid in the batch, or one origin for all grids
    void set_from_nearest_voxels_to_points(const JaggedTensor &points,
                                           const Vec3dBatchOrScalar &voxel_sizes = 1.0,
                                           const Vec3dBatch &origins             = torch::zeros(3,
                                                                                    torch::kInt32));

    /// @brief Populate the grid batch with the specified voxel coordinates
    /// @param ijk A JaggedTensor of shape [B, -1, 3] specifying the coordinates of each voxel to
    /// insert
    /// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid
    /// in the batch or one voxel size for all grids
    /// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the
    /// [0, 0, 0] voxel
    ///                for each grid in the batch, or one origin for all grids
    void set_from_ijk(const JaggedTensor &ijk,
                      const Vec3dBatchOrScalar &voxel_sizes = 1.0,
                      const Vec3dBatch &origins             = torch::zeros(3, torch::kInt32));

    /// @brief Populate the grid batch densely from ijk_min to ijk_min + size
    /// @param num_grids The number of grids to create in the batch
    /// @param dense_dims The size of each dense grid (shape [3,] = [W, H, D])
    /// @param ijk_min The minimum ijk coordinate of each dense grid in the batch (shape [3,])
    /// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid
    /// in the batch or one voxel size for all grids
    /// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the
    /// [0, 0, 0] voxel
    ///                for each grid in the batch, or one origin for all grids
    /// @param mask Optional mask of shape [W, H, D] to specify voxels which are included in the
    /// dense grid.
    ///             Note that the same mask will be re-used for all the grids in the batch.
    void set_from_dense_grid(const int64_t num_grids,
                             const Vec3i &dense_dims,
                             const Vec3i &ijk_min                  = torch::zeros(3, torch::kInt32),
                             const Vec3dBatchOrScalar &voxel_sizes = 1.0,
                             const Vec3dBatch &origins             = torch::zeros(3),
                             std::optional<torch::Tensor> mask     = std::nullopt);

    /// @brief Serialize this grid batch to a torch tensor of bytes (dtype = int8)
    /// @return A serialized grid batch encoded as a torch::Tensor of type int8
    torch::Tensor serialize() const;

    /// @brief Deserialize an int8 tensor (returned by serialize()) into a grid batch
    /// @param data A tensor enccoding a serialized grid batch as an int8 tensor
    /// @return The deserializes grid batch
    static GridBatch deserialize(const torch::Tensor &data);

    /// @brief Return an integer representing the actual data
    /// @return the value
    int64_t address() const;

    /// @brief Get the underlying nanovdb::GridHandle for the grid batch
    /// @return The underlying nanovdb::GridHandle for the grid batch
    const nanovdb::GridHandle<detail::TorchDeviceBuffer> &nanovdb_grid_handle() const;

    static GridBatch concatenate(const std::vector<GridBatch> &vec);

    static void computeConvolutionKernelMap(const GridBatch &source,
                                            const GridBatch &target,
                                            torch::Tensor &kernelMap,
                                            const Vec3iOrScalar &kernelSize,
                                            const Vec3iOrScalar &stride);

    std::vector<torch::Tensor> computeBrickHaloBuffer(bool benchmark) const;

  private:
    c10::intrusive_ptr<detail::GridBatchImpl> mImpl;
};

// using GridBatchPtr = c10::intrusive_ptr<GridBatch>;

} // namespace fvdb

#endif // FVDB_GRIDBATCH_H
