#pragma once
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>

#include <torch/script.h>
#include <torch/custom_class.h>

#include "detail/utils/Utils.h"
#include "detail/GridBatchImpl.h"

#include "JaggedTensor.h"
#include "Types.h"

namespace fvdb {



struct GridBatch : torch::CustomClassHolder {

    explicit GridBatch(TorchDeviceOrString device, bool isMutable);
    explicit GridBatch();

    GridBatch(c10::intrusive_ptr<detail::GridBatchImpl> gridHdl) : mImpl(gridHdl) {}

    /// @brief Return true if this is a contiguous view of the grid batch
    /// @return true if this is a contiguous view of the grid batch
    bool is_contiguous() const {
        return impl()->isContiguous();
    }

    /// @brief Return a contiguous copy of this grid batch. If the grid batch is already contiguous,
    ///        then return a reference to this
    /// @return A contiguous copy of this grid batch
    GridBatch contiguous() const {
        return GridBatch(detail::GridBatchImpl::contiguous(impl()));
    }

    /// @brief Get the voxel size of the bi^th grid in the batch and return is a tensor of type dtype
    /// @param bi The batch index of the grid for which to get the voxel size
    /// @param dtype The dtype of the returned tensor
    /// @return A tensor of shape [3,] containing the voxel size of the bi^th grid in the batch
    torch::Tensor voxel_size_at(uint32_t bi, const torch::Dtype& dtype = torch::kFloat32) const;

    /// @brief Get the voxel origin of the bi^th grid in the batch and return is a tensor of type dtype
    /// @param bi The batch index of the grid for which to get the voxel origin
    /// @param dtype The dtype of the returned tensor
    /// @return A tensor of shape [3,] containing the voxel origin of the bi^th grid in the batch
    torch::Tensor origin_at(uint32_t bi, const torch::Dtype& dtype = torch::kFloat32) const;

    /// @brief Get the voxel size of all grids in this batch and return is a tensor of type dtype
    /// @param dtype The dtype of the returned tensor
    /// @return A tensor of shape [grid_count(), 3] containing the voxel size of all grids indexed by this batch
    torch::Tensor voxel_sizes(const torch::Dtype& dtype = torch::kFloat32) const;

    /// @brief Get the voxel origins of all grids in this batch and return is a tensor of type dtype
    /// @param dtype The dtype of the returned tensor
    /// @return A tensor of shape [grid_count(), 3] containing the voxel origins of all grids indexed by this batch
    torch::Tensor origins(const torch::Dtype& dtype = torch::kFloat32) const;

    /// @brief Get the number of grids indexed by this batch
    /// @return The number of grids indexed by this batch
    int64_t grid_count() const {
        return impl()->batchSize();
    }

    /// @brief The total number of enabled voxels indexed by this batch of grids
    ///        For immutable grids, this returns the same value as total_voxels()
    /// @return The total number of enabled voxels indexed by this batch of grids
    int64_t total_enabled_voxels() const {
        return impl()->totalEnabledVoxels(false);
    }

    /// @brief Get the total number of voxels indexed by this batch of grids
    /// @return The total number of voxels indexed by this batch of grids
    int64_t total_voxels() const {
        return impl()->totalVoxels();
    }

    /// @brief Get the number of voxels indexed by the bi^th grid in the batch
    /// @param bid The batch index of the grid for which to get the number of voxels
    /// @return The number of voxels indexed by the bi^th grid in the batch
    int64_t num_voxels_at(uint32_t bid) const {
        return impl()->numVoxels(bid);
    }

    /// @brief Get the number of enabled voxels indexed by the bi^th grid in the batch.
    ///        For immutable grids, this returns the same value as num_voxels_at(bi)
    /// @param bid The batch index of the grid for which to get the number of enabled voxels
    /// @return The number of enabled voxels indexed by the bi^th grid in the batch
    int64_t num_enabled_voxels_at(uint32_t bid) const;

    /// @brief Get the cumulative number of voxels indexed by the first bid+1 grids
    /// @param bid The batch index
    /// @return The cumulative number of voxels indexed by the first bid+1 grids
    int64_t cum_voxels_at(uint32_t bid) const {
        return impl()->cumVoxels(bid);
    }

    /// @brief Get the cumulative number of enabled voxels indexed by the first bid+1 grids.
    ///        For immutable grids, this returns the same value as cum_voxels_at(bi)
    /// @param bid The batch index for which to get the cumulative number of enabled voxels
    /// @return The cumulative number of enabled voxels indexed by the first bid+1 grids
    int64_t cum_enabled_voxels_at(uint32_t bid) const;

    /// @brief Get the number of voxels per grid indexed by this batch of grids
    /// @return An integer tensor containing the number of voxels per grid indexed by this batch
    torch::Tensor num_voxels() const;

    /// @brief Get the number of enabled voxels indexed by this batch of grids
    ///        For immutable grids, this returns the same value as num_voxels()
    /// @return An integer tensor containing the number of enabled voxels per grid indexed by this batch
    torch::Tensor num_enabled_voxels() const;

    /// @brief Get the cumulative number of voxels indexed by the grids in this batch
    ///        i.e. [nvox_0, nvox_0+nvox_1, nvox_0+nvox_1+nvox_2, ...]
    /// @return An integer tensor containing the cumulative number of voxels indexed by the grids in this batch
    torch::Tensor cum_voxels() const;

    /// @brief Get the cumulative number of voxels indexed by the grids in this batch
    ///        i.e. [nvox_0, nvox_0+nvox_1, nvox_0+nvox_1+nvox_2, ...]
    /// @return An integer tensor containing the cumulative number of voxels indexed by the grids in this batch
    torch::Tensor cum_enabled_voxels() const;

    /// @brief Get the total number of bytes required to store all grids indexed by this batch
    /// @return The total number of bytes required to store all grids indexed by this batch
    int64_t total_bytes() const {
        return impl()->totalBytes();
    }

    /// @brief Get the number of bytes required to store each grid
    /// @return An integer tensor containing the number of bytes required to store each grid
    torch::Tensor num_bytes() const;

    /// @brief Get the total number of leaf nodes indexed by this batch of grids
    /// @return The total number of leaf nodes indexed by this batch of grids
    int64_t total_leaf_nodes() const {
        return impl()->totalLeaves();
    }

    /// @brief Get the number of leaf nodes in each grid
    /// @return An integer tensor containing the number of leaf nodes in each grid
    torch::Tensor num_leaf_nodes() const;


    /// @brief Get the offsets of the voxels indexed by this batch of grid
    /// @return A tensor of shape [batch_size, 2] where the [bi, 0]^th entry is the offset of the first voxel
    ///         and the [bi, 1]^th entry is the offset one past the last voxel indexed by the bi^th grid in the batch
    torch::Tensor joffsets() const {
        return impl()->voxelOffsets(true);
    }

    /// @brief Get the batch index for each voxel indexed by this batch of grids
    /// @return An int16 tensor of shape [total_voxels,] where the [i]^th entry is the batch index of the i^th voxel
    torch::Tensor jidx() const {
        torch::Tensor ret = impl()->jidx(true);
        if (grid_count() == 1 && ret.numel() == 0) {
            return torch::zeros({total_voxels()}, torch::TensorOptions().device(device()).dtype(torch::kInt16));
        } else {
            return ret;
        }

    }

    /// @brief Set the voxel size of all grids indexed by this batch to the specified value
    /// @param voxel_size A 3D (shape [3,]) tensor specifying the voxel size to set for each grid
    inline void set_global_voxel_size(const Vec3dOrScalar& voxel_size) {
        impl()->setGlobalVoxelSize(voxel_size.value());
    }

    /// @brief Set the voxel origin of all grids indexed by this batch to the specified value
    /// @param origin A 3D (shape [3,]) tensor specifying the voxel origin to set for each grid
    inline void set_global_origin(const Vec3d& origin) {
        impl()->setGlobalVoxelOrigin(origin.value());
    }

    /// @brief Return true if this grid is mutable
    /// @return Whether the grid is mutable
    inline bool is_mutable() const {
        return impl()->isMutable();
    }

    /// @brief Get the device on which this grid is stored
    /// @return The device on which this grid is stored
    inline c10::Device device() const {
        return impl()->device();
    }

    /// @brief Get the primal transforms of the grids in this batch (i.e. world to primal grid coordinates)
    /// @return A std::vector<VoxelCoordTransform> containing the primal transforms of the grids in this batch
    inline const std::vector<detail::VoxelCoordTransform> primal_transforms() const {
        std::vector<detail::VoxelCoordTransform> transforms;
        transforms.reserve(grid_count());
        for (uint32_t bi = 0; bi < grid_count(); ++bi) {
            transforms.push_back(primal_transform_at(bi));
        }
        return transforms;
    }

    /// @brief Get the dual transforms of the grids in this batch (i.e. world to dual grid coordinates)
    /// @return A std::vector<detail::VoxelCoordTransform> containing the dual transforms of the grids in this batch
    inline const std::vector<detail::VoxelCoordTransform> dual_transforms() const {
        std::vector<detail::VoxelCoordTransform> transforms;
        transforms.reserve(grid_count());
        for (uint32_t bi = 0; bi < grid_count(); ++bi) {
            transforms.push_back(dual_transform_at(bi));
        }
        return transforms;
    }

    /// @brief Get the primal transform of the bid^th grid in the batch (i.e. world to primal grid coordinates)
    /// @param bi The index of the grid in the batch for which to get the primal transform
    /// @return The primal transform of the bid^th grid in the batch
    inline const fvdb::detail::VoxelCoordTransform& primal_transform_at(uint32_t bi) const {
        return impl()->primalTransform(bi);
    }

    /// @brief Get the dual transform of the bid^th grid in the batch (i.e. world to dual grid coordinates)
    /// @param bi The index of the grid in the batch for which to get the dual transform
    /// @return The dual transform of the bid^th grid in the batch
    inline const fvdb::detail::VoxelCoordTransform& dual_transform_at(uint32_t bi) const {
        return impl()->dualTransform(bi);
    }

    /// @brief Get the bounding box (in voxel coordinates) for each grid in the batch
    /// @return A tensor bboxes of shape [B, 2, 3] where
    ///         bboxes[bi] = [[bmin_i, bmin_j, bmin_z=k], [bmax_i, bmax_j, bmax_k]] is the bi^th bounding box
    ///         such that bmin <= ijk < bmax for all voxels ijk in the bi^th grid
    const torch::Tensor bbox() const;

    /// @brief Get the bounding box (in voxel coordinates) of the bi^th grid in the batch
    /// @return A tensor, bbox, of shape [2, 3] where
    ///         bbox = [[bmin_i, bmin_j, bmin_z=k], [bmax_i, bmax_j, bmax_k]] is the bi^th bounding box
    ///         such that bmin <= ijk < bmax for all voxels ijk in the bi^th grid
    const torch::Tensor bbox_at(int64_t bi) const;

    /// @brief Get the bounding box (in voxel coordinates) for the dual of each grid in the batch
    /// @return A tensor bboxes of shape [B, 2, 3] where
    ///         bboxes[bi] = [[bmin_i, bmin_j, bmin_z=k], [bmax_i, bmax_j, bmax_k]] is the bi^th bounding box
    ///         such that bmin <= ijk < bmax for all voxels ijk in the dual of the bi^th grid
    const torch::Tensor dual_bbox() const;

    /// @brief Get the bounding box (in voxel coordinates) of the dual of the bi^th grid in the batch
    /// @return A tensor, bbox, of shape [2, 3] where
    ///         bbox = [[bmin_i, bmin_j, bmin_z=k], [bmax_i, bmax_j, bmax_k]] is the bi^th bounding box
    ///         such that bmin <= ijk < bmax for all voxels ijk in the dual of the bi^th grid
    const torch::Tensor dual_bbox_at(int64_t bi) const;

    /// @brief Get the bounding box (in voxel coordinates) which contains all the grids in this batch
    /// @return A tensor, total_bbox, of shape [2, 3] where
    ///         total_bbox = [[bmin_i, bmin_j, bmin_z=k], [bmax_i, bmax_j, bmax_k]] is the bounding box
    ///         such that bmin <= ijk < bmax for all voxels ijk in the batch
    const torch::Tensor total_bbox() const;

    /// @brief Downsample this batch of grids using maxpooling
    /// @param pool_factor How much to pool by (i,e, (2,2,2) means take max over 2x2x2 from start of window)
    /// @param data Data at each voxel in this grid to be downsampled (JaggedTensor of shape [B, -1, *])
    /// @param stride The stride to use when pooling
    /// @param coarse_grid An optional coarse grid used to specify the output. This is mainly used for memory
    ///                    efficiency so you can chache grids. If you don't pass it in, we'll just create it for you.
    /// @return A pair (coarseData, coarseGrid) where coarseData is a JaggedTensor of shape [B, -1, *] of downsampled data
    ///         and coarseGrid is a GridBatch representing the downsampled grid batch
    std::pair<JaggedTensor, GridBatch> max_pool(Vec3iOrScalar pool_factor,
                                                const JaggedTensor& data,
                                                Vec3iOrScalar stride = 0,
                                                torch::optional<GridBatch> coarse_grid = torch::nullopt) const;

    /// @brief Downsample this batch of grids using average pooling
    /// @param pool_factor How much to pool by (i,e, (2, 2, 2) means take max over 2x2x2 from start of window)
    /// @param data Data at each voxel in this grid to be downsampled (JaggedTensor of shape [B, -1, *])
    /// @param stride The stride to use when pooling
    /// @param coarse_grid An optional coarse grid used to specify the output. This is mainly used for memory
    ///                    efficiency so you can chache grids. If you don't pass it in, we'll just create it for you.
    /// @return A pair (coarseData, coarseGrid) where coarseData is a JaggedTensor of shape [B, -1, *] of downsampled data
    ///         and coarseGrid is a GridBatch representing the downsampled grid batch
    std::pair<JaggedTensor, GridBatch> avg_pool(Vec3iOrScalar pool_factor,
                                                const JaggedTensor& data,
                                                Vec3iOrScalar stride = 0,
                                                torch::optional<GridBatch> coarse_grid = torch::nullopt) const;

    /// @brief Subdivide this batch of grids using nearest neighbor interpolation
    /// @param subdiv_factor How much to upsample by (i,e, (2,2,2) means upsample by 2x2x2)
    /// @param data Data at each voxel in this grid to be upsampled (JaggedTensor of shape [B, -1, *])
    /// @param mask An optional mask of shape [B, -1] specifying which coarse voxels to upsample
    /// @param fine_grid An optional coarse grid used to specify the output. This is mainly used for memory
    ///                  efficiency so you can chache grids. If you don't pass it in, we'll just create it for you.
    /// @return A pair (fineData, fineGrid) where fineData is a JaggedTensor of shape [B, -1, *] of upsampled data and
    ///         fineGrid is a GridBatch representing the upsampled grid batch
    std::pair<JaggedTensor, GridBatch> subdivide(Vec3iOrScalar subdiv_factor,
                                                 const JaggedTensor& data,
                                                 const torch::optional<JaggedTensor> mask = torch::nullopt,
                                                 torch::optional<GridBatch> fine_grid = torch::nullopt) const;

    /// @brief Read the values from a dense tensor of the voxels at the specified coordinates
    /// @param dense_data A dense tensor of shape [B, W, H, D, *]
    /// @param dense_origins A tensor of shape [B, 3] or [3,] specifying the voxel coordinate(s) of the origin of the dense tensor i.e. [:, 0, 0, 0]
    /// @return A JaggedTensor with shape [B, -1, *] containing the values at the specified coordinates
    JaggedTensor read_from_dense(const torch::Tensor& dense_data,
                                 const Vec3iBatch& dense_origins = torch::zeros(3, torch::kInt32)) const;

    /// @brief Read the values from a JaggedTensor indexed by this batch into a dense tensor
    /// @param sparse_data A JaggedTensor of shape [B, -1, *] containing one value per voxel in the batch
    /// @param min_coord An optional minimum coordinate to read from the batch (in voxel coordinates).
    ///                  Defaults to the minimum coordinate of the batch.
    /// @param grid_size An optional grid size to read from the batch (in voxel coordinates).
    ///                  Defaults to the total size of a grid containing the whole batch.
    /// @return A dense tensor of shape [B, W, H, D, *] containing the values at the specified coordinates (and zero elsewhere)
    torch::Tensor read_into_dense(const JaggedTensor& sparse_data,
                                  const torch::optional<Vec3iBatch>& min_coord = torch::nullopt,
                                  const torch::optional<Vec3i>& grid_size = torch::nullopt) const;

    /// @brief Given a GridBatch and features associated with it,
    ///        return a JaggedTensor representing features for this batch of grid.
    ///        Fill any voxels not in the GridBatch with the default value.
    /// @param features A JaggedTensor of shape [B, -1, *] containing features associated with other_grid.
    /// @param other_grid A GridBatch representing the grid to fill from.
    /// @param default_value The value to fill in for voxels not in other_grid.
    JaggedTensor fill_to_grid(const JaggedTensor& features,
                              const GridBatch& other_grid,
                              float default_value = 0.0f) const;

    /// @brief Convert grid coordinates to world coordinates
    /// @param ijk A JaggedTensor of grid coordinates with shape [B, -1, 3] (one point set per grid in the batch)
    /// @return A JaggedTensor of world coordinates with shape [B, -1, 3] (one point set per grid in the batch)
    JaggedTensor grid_to_world(const JaggedTensor& ijk) const;

    /// @brief Convert world coordinates to grid coordinates
    /// @param xyz A JaggedTensor of world coordinates with shape [B, -1, 3] (one point set per grid in the batch)
    /// @return A JaggedTensor of grid coordinates with shape [B, -1, 3] (one point set per grid in the batch)
    JaggedTensor world_to_grid(const JaggedTensor& xyz) const;

    /// @brief Get grid-to-world matrices
    /// @return A JaggedTensor of grid-to-world matrices with shape [B, 4, 4]
    torch::Tensor grid_to_world_matrices(const torch::Dtype& dtype = torch::kFloat32) const;

    /// @brief Get world-to-grid matrices
    /// @return A JaggedTensor of world-to-grid matrices with shape [B, 4, 4]
    torch::Tensor world_to_grid_matrices(const torch::Dtype& dtype = torch::kFloat32) const;

    /// @brief Sample features on the grid batch using trilinear interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the batch)
    /// @param voxel_data a JaggedTensor of C-dimensional features at each voxel with shape [B, -1, C] or a Tensor of
    ///                   shape [N, C] where N is the total number of voxels in the batch
    ///                   (one item for each voxel in each grid in the batch)
    /// @return a JaggedTensor of sampled data with shape [B, -1, C] (one sample set per point)
    JaggedTensor sample_trilinear(const JaggedTensor& points,
                                  const JaggedTensor& voxel_data) const;

    /// @brief Sample features and spatial gradients on the grid batch using trilinear interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the batch)
    /// @param voxel_data a JaggedTensor of C-dimensional features at each voxel with shape [B, -1, C] or a Tensor of
    ///                   shape [N, C] where N is the total number of voxels in the batch
    ///                   (one item for each voxel in each grid in the batch)
    /// @return a pair (feat, grad_feat) which are JaggedTensors of sampled data with shape [B, -1, C], and [B, -1, C, 3]
    ///         respectively where feat are the sampled features and grad_feat are the spatial gradients of the sampled
    ///         features (one sample set per point)
    std::vector<JaggedTensor> sample_trilinear_with_grad(const JaggedTensor& points,
                                                         const JaggedTensor& voxel_data) const;

    /// @brief Sample features on the grid batch using bezier interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the batch)
    /// @param voxel_data a JaggedTensor of C-dimensional features at each voxel with shape [B, -1, C] or a Tensor of
    ///                   shape [N, C] where N is the total number of voxels in the batch
    ///                   (one item for each voxel in each grid in the batch)
    /// @return a JaggedTensor of sampled data with shape [B, -1, C] (one sample set per point)
    JaggedTensor sample_bezier(const JaggedTensor& points,
                               const JaggedTensor& voxel_data) const;

    /// @brief Sample features and spatial gradients on the grid batch using bezier interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the batch)
    /// @param voxel_data a JaggedTensor of C-dimensional features at each voxel with shape [B, -1, C] or a Tensor of
    ///                   shape [N, C] where N is the total number of voxels in the batch
    ///                   (one item for each voxel in each grid in the batch)
    /// @return a pair (feat, grad_feat) which are JaggedTensors of sampled data with shape [B, -1, C], and [B, -1, C, 3]
    ///         respectively where feat are the sampled features and grad_feat are the spatial gradients of the sampled
    ///         features (one sample set per point)
    std::vector<JaggedTensor> sample_bezier_with_grad(const JaggedTensor& points,
                                                      const JaggedTensor& voxel_data) const;

    /// @brief Splat features at points into a grid batch using trilinear interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the batch)
    /// @param points_data a JaggedTensor of C-dimensional features at each point with shape [B, -1, C]
    /// @return a JaggedTensor of C-dimensional features at each voxel in the batch with shape [B, -1, C]
    JaggedTensor splat_trilinear(const JaggedTensor& points,
                                 const JaggedTensor& points_data) const;

    /// @brief Splat features at points into a grid using bezier interpolation
    /// @param points a JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the batch)
    /// @param points_data a JaggedTensor of C-dimensional features at each point with shape [B, -1, C]
    /// @return a JaggedTensor of C-dimensional features at each voxel in the batch with shape [B, -1, C]
    JaggedTensor splat_bezier(const JaggedTensor& points,
                              const JaggedTensor& points_data) const;

    /// @brief Get the indices of neighbors in the N-ring of each voxel in the grid batch
    ///        (possibly bitshifting the coordinates which is useful when you use multiple grids to represent different
    ///        levels of a hierarchy and you want to query this grid with coordinates at a finer level)
    /// @param ijk A JaggedTensor of voxel coordinates with shape [B, -1, 3] (one set of coordinates per grid in the batch)
    /// @param extent The size of a neighborhood to find indexes
    /// @param bitshift The number of bits to shift the coordinates by
    /// @return A JaggedTensor of neighbor indexes with shape [B, -1, 2*extent+1, 2*extent+1, 2*extent+1] (-1 value indicates no neighbor at that index)
    JaggedTensor neighbor_indexes(const JaggedTensor& ijk,
                                  int32_t extent,
                                  int32_t bitshift = 0) const;

    /// @brief Return whether each point lies inside the grid batch
    /// @param xyz A JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the batch)
    /// @param ignore_disabled Whether to ignore voxels that have been disabled (only applicable to mutable grids)
    /// @return A JaggedTensor of booleans with shape [B, -1] (one boolean per point)
    ///         where the [bi, i]^th entry is true if points[bi, i] lies inside the bi^th grid in the batch
    JaggedTensor points_in_active_voxel(const JaggedTensor& xyz,
                                        bool ignore_disabled = false) const;

    /// @brief Return whether the cube with corners at cube_min and cube_max centered at each point in world space
    ///        intersect the grid batch
    /// @param cube_centers A JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the batch)
    /// @param cube_min A 3D tensor specifying the min corner relative to each point to check
    /// @param cube_max A 3D tensor specifying the max corner relative to each point to check
    /// @param ignore_disabled Whether to ignore voxels that have been disabled (only applicable to mutable grids)
    /// @return A JaggedTensor of booleans with shape [B, -1] (one boolean per point)
    ///         where the [bi, i]^th entry is true if the cube with extent (min, max) + points[bi, i] intersects
    ///         the bi^th grid in the batch
    JaggedTensor cubes_intersect_grid(const JaggedTensor& cube_centers,
                                      const Vec3dOrScalar& cube_min = 0.0,
                                      const Vec3dOrScalar& cube_max = 0.0,
                                      bool ignore_disabled = false) const;

    /// @brief Return whether the cube with corners at cube_min and cube_max centered at each point in world space
    ///        is fully contained in the grid batch's stencil
    /// @param cube_centers A JaggedTensor of points with shape [B, -1, 3] (one point set per grid in the batch)
    /// @param cube_min A 3D tensor specifying the min corner relative to each point to check
    /// @param cube_max A 3D tensor specifying the max corner relative to each point to check
    /// @param ignore_disabled Whether to ignore voxels that have been disabled (only applicable to mutable grids)
    /// @return A JaggedTensor of booleans with shape [B, -1] (one boolean per point)
    ///         where the [bi, i]^th entry is true if the cube with extent (min, max) + points[bi, i] lies
    ///         inside the bi^th grid in the batch
    JaggedTensor cubes_in_grid(const JaggedTensor& cube_centers,
                               const Vec3dOrScalar& cube_min = 0.0,
                               const Vec3dOrScalar& cube_max = 0.0,
                               bool ignore_disabled = false) const;

    /// @brief Return a boolean mask indicating whether each voxel in the grid is enabled or not
    /// @return A boolean JaggedTensor of shape [B, -1] indicating whether each voxel in the grid is enabled or not
    JaggedTensor enabled_mask() const;

    /// @brief Return a boolean mask indicating whether each voxel in the grid is disabled or not
    /// @return A boolean JaggedTensor of shape [B, -1] indicating whether each voxel in the grid is disabled or not
    JaggedTensor disabled_mask() const;

    /// @brief Return whether each coordinate is in the grid batch or not
    /// @param ijk A JaggedTensor of coordinates with shape [B, -1, 3] (one coordinate set per grid in the batch)
    /// @param ignore_disabled Whether to ignore voxels that have been disabled (only applicable to mutable grids)
    /// @return A JaggedTensor of booleans with shape [B, -1] (one boolean per coordinate)
    ///         where the [bi, i]^th entry is true if coords[bi, i] lies inside the bi^th grid in the batch
    JaggedTensor coords_in_active_voxel(const JaggedTensor& ijk, bool ignore_disabled = false) const;

    /// @brief Return the integer offset of each ijk value in the grid batch
    /// @param ijk A JaggedTensor of ijk coordinates with shape [B, -1, 3] (one coordinate set per grid in the batch)
    /// @return A JaggedTensor of integer offsets with shape [B, -1] into the grid batch (one offset per coordinate)
    JaggedTensor ijk_to_index(const JaggedTensor& ijk) const;

    /// @brief Return a JaggedTensor of integers such that if it is used as a permutation of the input IJK coordinates,
    ///        it will re-order them to the indexing order of the grid batch. This effectively performs the inverse of
    ///        ijkToIndex if you pass in the ijk coordinates in the grid.
    /// @param ijk A JaggedTensor of ijk coordinates with shape [B, -1, 3] (one coordinate set per grid in the batch)
    /// @return A JaggedTensor of integers with shape [B, -1] (one integer per grids' ijk) which inverts ijkToIndex
    JaggedTensor ijk_to_inv_index(const JaggedTensor& ijk) const;

    /// @brief Return the set of active ijk coordinates indexed by this grid batch
    /// @return A JaggedTensor of voxel coordinates indexed by this grid batch (shape [B, -1, 3])
    JaggedTensor ijk() const;

    /// @brief Return the set of enabled ijk coordinates indexed by this grid batch
    /// @note For non mutable grids, this function returns the same values as ijk()
    /// @return A JaggedTensor of voxel coordinates indexed by this grid batch (shape [B, -1, 3])
    JaggedTensor ijk_enabled() const;

    /// @brief Find the intersection between a collection of rays and the zero level set of a scalar field
    ///        at each voxel in the grid batch
    /// @param ray_origins A JaggedTensor of ray origins with shape [B, -1, 3] (one ray set per grid in the batch)
    /// @param ray_directions A JaggedTensor of ray directions with shape [B, -1, 3] (one ray set per grid in the batch)
    /// @param grid_scalars A JaggedTensor of scalar values with shape [B, -1] (one scalar per voxel in the batch)
    /// @param eps Skip voxels where the ray intersects by less than this distance
    /// @return A JaggedTensor of intersection times with shape [B, -1] (one time per ray)
    JaggedTensor ray_implicit_intersection(const JaggedTensor& ray_origins,
                                           const JaggedTensor& ray_directions,
                                           const JaggedTensor& grid_scalars,
                                           double eps = 0.0) const;

    std::vector<JaggedTensor> voxels_along_rays(const JaggedTensor& ray_origins,
                                                const JaggedTensor& ray_directions,
                                                int64_t max_voxels, double eps = 0.0) const;

    std::vector<JaggedTensor> segments_along_rays(const JaggedTensor& ray_origins,
                                                  const JaggedTensor& ray_directions,
                                                  int64_t max_segments, double eps = 0.0, bool ignore_masked = false) const;

    std::vector<JaggedTensor> uniform_ray_samples(const JaggedTensor& ray_origins,
                                                  const JaggedTensor& ray_directions,
                                                  const JaggedTensor& t_min,
                                                  const JaggedTensor& t_max,
                                                  double step_size,
                                                  double cone_angle = 0.0,
                                                  bool include_end_segments = true) const;

    /// @brief Return an edge network used which can be used to plot the grids in this batch
    /// @param return_voxel_coordinates Whether to return the vertices in voxel coordinates or world coordinates
    /// @return A pair (verts, edges) where verts is a JaggedTensor of vertex positions with shape [B, -1, 3]
    ///         (one vertex set per grid in the batch) and edges is a JaggedTensor of edge indices of
    ///         shape [B, -1, 2] (one edge set per grid in the batch)
    std::vector<JaggedTensor> viz_edge_network(bool return_voxel_coordinates = false) const;

    /// @brief Disable the specified voxels in the grid batch. If the input ijk values refer to non-indexed voxels,
    ///        then these are simply ignored.
    /// @param ijk A Jagged tensor of shape [B, -1, 3] of coordinates to disable(one set of coordinates per grid in the batch)
    /// @note This is only applicable to mutable grids
    void disable_ijk(const JaggedTensor& ijk);

    /// @brief Enable the specified voxels in the grid batch. If the input ijk values refer to non-indexed voxels,
    ///        then these are simply ignored.
    /// @param ijk A Jagged tensor of shape [B, -1, 3] of coordinates to enable (one set of coordinates per grid in the batch)
    /// @note This is only applicable to mutable grids
    void enable_ijk(const JaggedTensor& ijk);

    /// @brief Return a batch of grids representing the dual of this batch. i.e. The centers of the dual grid correspond
    ///        to the corners of this grid batch. The [i, j, k] coordinate of the dual grid corresponds to the bottom/left/back
    ///        corner of the [i, j, k] voxel in this grid batch.
    /// @param exclude_border Whether to exclude the border of the grid batch when computing the dual grid
    /// @return A GridBatch representing the dual of this grid batch
    GridBatch dual_grid(bool exclude_border = false) const;

    /// @brief Return a batch of grids representing the coarsened version of this batch.
    ///        Each voxel [i, j, k] in this grid batch maps to voxel [i / branchFactor, j / branchFactor, k / branchFactor]
    ///        in the coarse batch.
    /// @param coarsening_factor The factor by which to coarsen the grid batch (i.e (2, 2, 2) coarses by a factor of 2x2x2)
    /// @return A GridBatch representing the coarsened version of this batch.
    GridBatch coarsened_grid(Vec3iOrScalar coarsening_factor) const;

    /// @brief Subdivide the grid batch into a finer grid batch.
    ///        Each voxel [i, j, k] in this grid batch maps to voxels [i * subdivFactor, j * subdivFactor, k * subdivFactor]
    ///        in the fine batch.
    /// @param subdiv_factor The factor by which to subdivide the grid batch
    /// @param mask An optional JaggedTensor of shape [B, -1] of boolean values indicating which voxels to subdivide
    /// @return A GridBatch representing the subdivided version of this batch.
    GridBatch subdivided_grid(Vec3iOrScalar subdiv_factor,
                              const torch::optional<JaggedTensor> mask = torch::nullopt) const;

    /// @brief Return a batch of grids representing the clipped version of this batch of grids.
    /// @param ijk_min Index space minimum bound of the clip region.
    /// @param ijk_max Index space maximum bound of the clip region.
    /// @return A GridBatch representing the clipped version of this batch of grids.
    GridBatch clipped_grid(const Vec3iBatch& ijk_min, const Vec3iBatch& ijk_max) const;

    /// @brief Generate the grid that is affected by the convolution operator.
    /// @param kernel_size The kernel size of convolution
    /// @param stride The stride of the convolution
    /// @return A GridBatch representing the convolved grid.
    GridBatch conv_grid(Vec3iOrScalar kernel_size, Vec3iOrScalar stride) const;

    /// @brief Return a batch of grids representing the clipped version of this batch of grids and corresponding features.
    /// @param features A JaggedTensor of shape [B, -1, *] containing features associated with this batch of grids.
    /// @param ijk_min Index space minimum bound of the clip region.
    /// @param ijk_max Index space maximum bound of the clip region.
    /// @return A pair (clipped_features, clipped_grid) where clipped_features is a JaggedTensor of shape [B, -1, *] and
    ///         clipped_grid is a GridBatch representing the clipped version of this batch of grids.
    std::pair<JaggedTensor, GridBatch> clip(const JaggedTensor& features, const Vec3iBatch& ijk_min,  const Vec3iBatch& ijk_max) const;

    /// @brief Extract 0-isosurface from an implicit field.
    /// @param field implicit value stored on each voxel center (or voxel corner on a dual grid)
    /// @param level level set of the surface to extract
    /// @return vertices and faces arrays of the extracted isosurface
    std::vector<JaggedTensor> marching_cubes(const JaggedTensor& field, double level = 0.0) const;

    /// @brief Perform in-grid convolution using fast halo buffer method. Currently only supports kernel_size = 3.
    /// @param features A JaggedTensor of shape [B, -1, *] containing features associated with this batch of grids.
    /// @param kernel A tensor of shape [Out, In, 3, 3, 3] containing the kernel to convolve with.
    /// @return A JaggedTensor of shape [B, -1, *] containing the convolved features.
    JaggedTensor sparse_conv_halo(const JaggedTensor& features, const torch::Tensor& kernel, int variant) const;

    /// @brief Return a grid batch on the specified device. If the passed in device is the same as this grid batch's
    ///        device, then this grid batch is returned. Otherwise, a copy of this grid batch is returned on the specified device.
    /// @param to_device The device to return the grid batch on
    /// @return A GridBatch representing this grid batch on the specified device
    GridBatch to(TorchDeviceOrString to_device) const {
        torch::Device toDevice = to_device.value();
        if (toDevice == device()) {
            return GridBatch(impl());
        } else {
            return GridBatch(impl()->clone(toDevice));
        }
    }

    /// @brief Return a grid batch on the same device as the specified grid batch. If the passed in grid has the same device as this grid batch's
    ///        device, then this grid batch is returned. Otherwise, a copy of this grid batch is returned on the specified device.
    /// @param to_grid The grid batch used to specify which device to return the grid batch on
    /// @return A GridBatch representing this grid batch on the specified device
    GridBatch to(const GridBatch& to_grid) const {
        return this->to(to_grid.device());
    }

    /// @brief Return a grid batch on the same device as the specified tensor. If the passed in tensor has the same device as this grid batch's
    ///        device, then this grid batch is returned. Otherwise, a copy of this grid batch is returned on the specified device.
    /// @param to_tensor The tensor used to specify which device to return the grid batch on
    /// @return A GridBatch representing this grid batch on the specified device
    GridBatch to(const torch::Tensor& to_tensor) const {
        return this->to(to_tensor.device());
    }

    /// @brief Return a grid batch on the same device as the specified JaggedTensor. If the passed in JaggedTensor has the same device as this grid batch's
    ///        device, then this grid batch is returned. Otherwise, a copy of this grid batch is returned on the specified device.
    /// @param to_jtensor The JaggedTensor used to specify which device to return the grid batch on
    /// @return A GridBatch representing this grid batch on the specified device
    GridBatch to(const JaggedTensor& to_jtensor) const {
        return this->to(to_jtensor.device());
    }

    /// @brief Return a view of this grid batch containing the grid at the specified index i.e. grid_batch[bid]
    /// @param bid The index to get a view on
    /// @return A GridBatch representing the grid at the specified index
    GridBatch index(int32_t bid) const {
        return GridBatch(impl()->index(bid));
    }

    /// @brief Return a slice view of this grid batch i.e. grid_batch[start:stop:step]
    /// @param start The start index of the slice
    /// @param stop The stop index of the slice
    /// @param step The step of the slice
    /// @return A GridBatch representing the slice of this grid batch
    GridBatch index(size_t start, size_t stop, size_t step) const {
        return GridBatch(impl()->index(start, stop, step));
    }

    /// @brief Return a view of this grid batch at the specified indices i.e. grid_batch[[i1, i2, ...]]
    /// @param idx A list of integers representing the indices to get a view on
    /// @return The grid batch vieweed at the specified indices
    GridBatch index(const std::vector<int64_t>& idx) const {
        return GridBatch(impl()->index(idx));
    }

    /// @brief Return a view of this grid batch at indices specified by the given mask i.e. grid_batch[mask]
    /// @param idx A list of integers representing the indices to get a view on
    /// @return The grid batch vieweed at the specified indices
    GridBatch index(const std::vector<bool>& idx) const {
        return GridBatch(impl()->index(idx));
    }

    /// @brief Return a view of this grid batch at the specified indices (or mask if idx is a bool tensor) i.e. grid_batch[[i1, i2, ...]]
    /// @param idx A list of integers representing the indices to get a view on
    /// @return The grid batch vieweed at the specified indices
    GridBatch index(const torch::Tensor& idx) const {
        return GridBatch(impl()->index(idx));
    }

    /// @brief Return a JaggedTensor whose joffsets and jidx match this grid batch's
    /// @param data The data to use for the JaggedTensor (first dimension must match the total number of voxels in the grid batch)
    /// @param ignore_disabled If true, then voxels which are disabled will be included in the returned JaggedTensor
    /// @return A JaggedTensor corresponding to the voxel grid of this grid batch
    JaggedTensor jagged_like(const torch::Tensor& data, bool ignore_disabled = true) const {
        return impl()->jaggedTensor(data, ignore_disabled);
    }

    /// @brief Populate the grid batch with voxels that intersect a triangle mesh
    /// @param vertices A JaggedTensor of shape [B, -1, 3] containing one vertex set per grid to create
    /// @param faces  A JaggedTensor of shape [B, -1, 3] containing one face set per grid to create
    /// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in the batch or one voxel size for all grids
    /// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0, 0, 0] voxel
    ///                for each grid in the batch, or one origin for all grids
    void set_from_mesh(const JaggedTensor& vertices,
                       const JaggedTensor& faces,
                       const Vec3dBatchOrScalar& voxel_sizes = 1.0,
                       const Vec3dBatch& origins = torch::zeros(3, torch::kInt32));

    /// @brief Populate the grid batch with voxels which contain a point in an input set of point clouds
    ///        (possibly padding each voxel containing a point)
    /// @param points A JaggedTensor with shape [B, -1, 3] containing one point set per grid to create
    /// @param pad_min A tensor of shape [3,] containing the number of voxels to pad each inserted voxel with to the left/back/bottom
    /// @param pad_max A tensor of shape [3,] containing the number of voxels to pad each inserted voxel with to the right/front/top
    /// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in the batch or one voxel size for all grids
    /// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0, 0, 0] voxel
    ///                for each grid in the batch, or one origin for all grids
    /// @param isMutable Whether the grid should be mutable or not
    void set_from_points(const JaggedTensor& points,
                         const Vec3i& pad_min = torch::zeros(3, torch::kInt32),
                         const Vec3i& pad_max = torch::zeros(3, torch::kInt32),
                         const Vec3dBatchOrScalar& voxel_sizes = 1.0,
                         const Vec3dBatch& origins = torch::zeros(3, torch::kInt32));

    /// @brief Populate the grid batch with the eight nearest voxels to each point in an input set of point clouds
    /// @param points A JaggedTensor with shape [B, -1, 3] containing one point set per grid to create
    /// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in the batch or one voxel size for all grids
    /// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0, 0, 0] voxel
    ///                for each grid in the batch, or one origin for all grids
    /// @param isMutable Whether the grid should be mutable or not
    void set_from_nearest_voxels_to_points(const JaggedTensor& points,
                                           const Vec3dBatchOrScalar& voxel_sizes = 1.0,
                                           const Vec3dBatch& origins = torch::zeros(3, torch::kInt32));


    /// @brief Populate the grid batch with the specified voxel coordinates (possibly with padding)
    /// @param ijk A JaggedTensor of shape [B, -1, 3] specifying the coordinates of each voxel to insert
    /// @param pad_min A tensor of shape [3,] containing the number of voxels to pad each inserted voxel with to the left/back/bottom
    /// @param pad_max A tensor of shape [3,] containing the number of voxels to pad each inserted voxel with to the right/front/top
    /// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in the batch or one voxel size for all grids
    /// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0, 0, 0] voxel
    ///                for each grid in the batch, or one origin for all grids
    void set_from_ijk(const JaggedTensor& ijk,
                      const Vec3i& pad_min = torch::zeros(3, torch::kInt32),
                      const Vec3i& pad_max = torch::zeros(3, torch::kInt32),
                      const Vec3dBatchOrScalar& voxel_sizes = 1.0,
                      const Vec3dBatch& origins = torch::zeros(3, torch::kInt32));

    /// @brief Populate the grid batch densely from ijk_min to ijk_min + size
    /// @param num_grids The number of grids to create in the batch
    /// @param dense_dims The size of each dense grid (shape [3,] = [W, H, D])
    /// @param ijk_min The minimum ijk coordinate of each dense grid in the batch (shape [3,])
    /// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in the batch or one voxel size for all grids
    /// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0, 0, 0] voxel
    ///                for each grid in the batch, or one origin for all grids
    /// @param mask Optional mask of shape [W, H, D] to specify voxels which are included in the dense grid.
    ///             Note that the same mask will be re-used for all the grids in the batch.
    void set_from_dense_grid(const int64_t num_grids,
                             const Vec3i& dense_dims,
                             const Vec3i& ijk_min = torch::zeros(3, torch::kInt32),
                             const Vec3dBatchOrScalar& voxel_sizes = 1.0,
                             const Vec3dBatch& origins = torch::zeros(3),
                             torch::optional<torch::Tensor> mask = torch::nullopt);

    /// @brief Serialize this grid batch to a torch tensor of bytes (dtype = int8)
    /// @return A serialized grid batch encoded as a torch::Tensor of type int8
    torch::Tensor serialize() const {
        return impl()->serialize();
    }

    /// @brief Deserialize an int8 tensor (returned by serialize()) into a grid batch
    /// @param data A tensor enccoding a serialized grid batch as an int8 tensor
    /// @return The deserializes grid batch
    static GridBatch deserialize(const torch::Tensor& data) {
        return GridBatch(detail::GridBatchImpl::deserialize(data));
    }

    /// @brief Return an integer representing the actual data
    /// @return the value
    int64_t address() const {
        return reinterpret_cast<int64_t>(impl().get());
    }

    /// @brief Get the underlying nanovdb::GridHandle for the grid batch
    /// @return The underlying nanovdb::GridHandle for the grid batch
    const nanovdb::GridHandle<detail::PytorchDeviceBuffer>& nanovdb_grid_handle() const {
        return impl()->nanoGridHandle();
    }

    inline const c10::intrusive_ptr<detail::GridBatchImpl> impl() const {
        return mImpl;
    }

private:

    void buildCoarseFromFineGrid(const GridBatch& fineGrid, nanovdb::Coord branchFactor);

    void buildFineFromCoarseGrid(const GridBatch& coarseGrid, const torch::optional<JaggedTensor>& subdivMask, nanovdb::Coord subdivFactor);

    void buildDualFromPrimalGrid(const GridBatch& primalGrid, bool excludeBorder = false);

    c10::intrusive_ptr<detail::GridBatchImpl> mImpl;
};


// using GridBatchPtr = c10::intrusive_ptr<GridBatch>;

} // namespace fvdb
