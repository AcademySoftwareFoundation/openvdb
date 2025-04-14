// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_FVDB_H
#define FVDB_FVDB_H

#include "GridBatch.h"
#include "JaggedTensor.h"
#include "SparseConvPackInfo.h"
#include "Types.h"

#include <torch/all.h>

namespace fvdb {

std::vector<torch::Tensor> volumeRender(const torch::Tensor &sigmas, const torch::Tensor &rgbs,
                                        const torch::Tensor &deltaTs, const torch::Tensor &ts,
                                        const torch::Tensor &packInfo, double transmittanceThresh);

JaggedTensor scaledDotProductAttention(const JaggedTensor &query, const JaggedTensor &key,
                                       const JaggedTensor &value, float scale);

/// @brief Concatenate a list of grid batches into a single grid batch
/// @param vec A list of grid batches to concatenate
/// @return A GridBatch representing the concatenated grid batch
GridBatch jcat(const std::vector<GridBatch> &vec);

/// @brief Concatenate a list of JaggedTensor into a single JaggedTensor
/// @param vec A list of JaggedTensor to concatenate
/// @param dim The dimension to concatenate along or nullptr to concatenate the outermost tensor
/// lists
/// @return A JaggedTensor representing the concatenated JaggedTensor
JaggedTensor jcat(const std::vector<JaggedTensor> &vec, std::optional<int64_t> dim = std::nullopt);

/// @brief Create a JaggedTensor filled with random numbers from a uniform distribution
///        on the interval [0, 1) with the specified lshape an rshape
/// @param lsizes The lshape of the JaggedTensor (number of elements per tensor)
/// @param rsizes The rshape of the JaggedTensor (feature dimension of each tensor)
/// @param options The options to use for the created tensor
/// @return A JaggedTensor filled with random numbers from the uniform distribution on [0, 1).
JaggedTensor jrand(const std::vector<int64_t> &lsizes, const std::vector<int64_t> rsizes = {},
                   at::TensorOptions options = {});
JaggedTensor jrand(const std::vector<std::vector<int64_t>> &lsizes,
                   const std::vector<int64_t> rsizes = {}, at::TensorOptions options = {});

/// @brief Create a JaggedTensor filled with random numbers from a normal distribution
///        with mean 0 and variance 1 (also called the standard normal distribution).
/// @param lsizes The lshape of the JaggedTensor (number of elements per tensor)
/// @param rsizes The rshape of the JaggedTensor (feature dimension of each tensor)
/// @param options The options to use for the created tensor
/// @return A JaggedTensor filled with random numbers from the standard normal distribution.
JaggedTensor jrandn(const std::vector<int64_t> &lsizes, const std::vector<int64_t> rsizes = {},
                    at::TensorOptions options = {});
JaggedTensor jrandn(const std::vector<std::vector<int64_t>> &lsizes,
                    const std::vector<int64_t> rsizes = {}, at::TensorOptions options = {});

/// @brief Create a JaggedTensor filled with zeros.
/// @param lsizes The lshape of the JaggedTensor (number of elements per tensor)
/// @param rsizes The rshape of the JaggedTensor (feature dimension of each tensor)
/// @param options The options to use for the created tensor
/// @return A JaggedTensor filled with zeros.
JaggedTensor jzeros(const std::vector<int64_t> &lsizes, const std::vector<int64_t> rsizes = {},
                    at::TensorOptions options = {});
JaggedTensor jzeros(const std::vector<std::vector<int64_t>> &lsizes,
                    const std::vector<int64_t> rsizes = {}, at::TensorOptions options = {});

/// @brief Create a JaggedTensor filled with ones.
/// @param lsizes The lshape of the JaggedTensor (number of elements per tensor)
/// @param rsizes The rshape of the JaggedTensor (feature dimension of each tensor)
/// @param options The options to use for the created tensor
/// @return A JaggedTensor filled with ones.
JaggedTensor jones(const std::vector<int64_t> &lsizes, const std::vector<int64_t> rsizes = {},
                   at::TensorOptions options = {});
JaggedTensor jones(const std::vector<std::vector<int64_t>> &lsizes,
                   const std::vector<int64_t> rsizes = {}, at::TensorOptions options = {});

/// @brief Create an empty JaggedTensor with uninitialized values.
/// @param lsizes The lshape of the JaggedTensor (number of elements per tensor)
/// @param rsizes The rshape of the JaggedTensor (feature dimension of each tensor)
/// @param options The options to use for the created tensor
/// @return A JaggedTensor filled with uninitialized values.
JaggedTensor jempty(const std::vector<int64_t> &lsizes, const std::vector<int64_t> rsizes = {},
                    at::TensorOptions options = {});
JaggedTensor jempty(const std::vector<std::vector<int64_t>> &lsizes,
                    const std::vector<int64_t> rsizes = {}, at::TensorOptions options = {});

/// @brief Return a grid batch with voxels which contain a point in an input set of point clouds
///        (possibly padding each voxel containing a point)
/// @param points A JaggedTensor with shape [B, -1, 3] containing one point set per grid to create
/// @param pad_min A tensor of shape [3,] containing the number of voxels to pad each inserted voxel
/// with to the left/back/bottom
/// @param pad_max A tensor of shape [3,] containing the number of voxels to pad each inserted voxel
/// with to the right/front/top
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel
///                for each grid in the batch, or one origin for all grids
/// @param is_mutable Whether the grid should be mutable or not
/// @return A GridBatch containing the created grid batch
GridBatch gridbatch_from_points(const JaggedTensor &points,
                                const Vec3i        &pad_min = torch::zeros({ 3 }, torch::kInt32),
                                const Vec3i        &pad_max = torch::zeros({ 3 }, torch::kInt32),
                                const Vec3dBatchOrScalar &voxel_sizes = 1.0,
                                const Vec3dBatch         &origins     = torch::zeros({ 3 }),
                                bool                      is_mutable  = false);

/// @brief Return a grid batch with the eight nearest voxels to each point in an input set of point
/// clouds
/// @param points A JaggedTensor with shape [B, -1, 3] containing one point set per grid to create
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel
///                     for each grid in the batch, or one origin for all grids
/// @param is_mutable Whether the grid should be mutable or not
/// @return A GridBatch containing the created grid batch
GridBatch gridbatch_from_nearest_voxels_to_points(const JaggedTensor       &points,
                                                  const Vec3dBatchOrScalar &voxel_sizes = 1.0,
                                                  const Vec3dBatch &origins = torch::zeros({ 3 }),
                                                  bool              is_mutable = false);

/// @brief Return a grid batch with the specified voxel coordinates (possibly with padding)
/// @param coords A JaggedTensor of shape [B, -1, 3] specifying the coordinates of each voxel to
/// insert
/// @param pad_min A tensor of shape [3,] containing the number of voxels to pad each inserted voxel
/// with to the left/back/bottom
/// @param pad_max A tensor of shape [3,] containing the number of voxels to pad each inserted voxel
/// with to the right/front/top
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel
///                for each grid in the batch, or one origin for all grids
/// @return A GridBatch containing the created grid batch
GridBatch gridbatch_from_ijk(const JaggedTensor       &ijk,
                             const Vec3i              &pad_min = torch::zeros({ 3 }, torch::kInt32),
                             const Vec3i              &pad_max = torch::zeros({ 3 }, torch::kInt32),
                             const Vec3dBatchOrScalar &voxel_sizes = 1.0,
                             const Vec3dBatch         &origins     = torch::zeros({ 3 }),
                             bool                      is_mutable  = false);

/// @brief Return a grid batch densely from ijkMin to ijkMin + size
/// @param numGrids The number of grids to create in the batch
/// @param denseDims The size of each dense grid (shape [3,] = [W, H, D])
/// @param ijkMin The minimum ijk coordinate of each dense grid in the batch (shape [3,])
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel
///                     for each grid in the batch, or one origin for all grids
/// @param mask Optional mask of shape [W, H, D] to specify voxels which are included in the dense
/// grid.
///             Note that the same mask will be re-used for all the grids in the batch.
/// @param device Which device to build the grid batch on
/// @param mutable If the returned grid batch should be mutable
/// @return A GridBatch containing a batch of dense grids
GridBatch gridbatch_from_dense(const int64_t numGrids, const Vec3i &denseDims, const Vec3i &ijkMin,
                               const Vec3dBatchOrScalar    &voxel_sizes = 1.0,
                               const Vec3dBatch            &origins     = torch::zeros({ 3 }),
                               std::optional<torch::Tensor> mask        = std::nullopt,
                               const torch::Device &device = torch::kCPU, bool is_mutable = false);

/// @brief Return a grid batch densely from ijkMin to ijkMin + size
/// @param numGrids The number of grids to create in the batch
/// @param denseDims The size of each dense grid (shape [3,] = [W, H, D])
/// @param ijkMin The minimum ijk coordinate of each dense grid in the batch (shape [3,])
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel
///                     for each grid in the batch, or one origin for all grids
/// @param mask Optional mask of shape [W, H, D] to specify voxels which are included in the dense
/// grid.
///             Note that the same mask will be re-used for all the grids in the batch.
/// @param device String specifying which device to build the grid batch on
/// @param mutable If the returned grid batch should be mutable
/// @return A GridBatch containing a batch of dense grids
GridBatch gridbatch_from_dense(const int64_t numGrids, const Vec3i &denseDims, const Vec3i &ijkMin,
                               const Vec3dBatchOrScalar    &voxel_sizes = 1.0,
                               const Vec3dBatch            &origins     = torch::zeros({ 3 }),
                               std::optional<torch::Tensor> mask        = std::nullopt,
                               const std::string &device_string = "cpu", bool is_mutable = false);

/// @brief Return a grid batch from a jagged batch of triangle meshes (i.e. each voxel intersects
/// the mesh)
/// @param vertices A JaggedTensor of shape [B, -1, 3] containing the vertices of each mesh in the
/// batch
/// @param faces A JaggedTensor of shape [B, -1, 3] containing the faces of each mesh in the batch
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel
///                for each grid in the batch, or one origin for all grids
/// @param is_mutable Whether the grid should be mutable or not
/// @return A GridBatch containing the created grid batch
GridBatch gridbatch_from_mesh(const JaggedTensor &vertices, const JaggedTensor &faces,
                              const Vec3dBatchOrScalar &voxel_sizes, const Vec3dBatch &origins,
                              bool is_mutable);

/// @brief Return a grid batch, tensors of data, and names from a nanovdb grid handle
/// @param handle nanovdb grid handle
/// @return A triple (gridbatch, data, names) where gridbatch is a GridBatch containing the
/// converted grids,
///         data is a JaggedTensor containing the data of the grids, and names is a list of strings
///         containing the name of each grid
std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
from_nanovdb(nanovdb::GridHandle<nanovdb::HostBuffer> &handle);

/// @brief Return a nanovdb grid handle created from a grid batch, optional jagged tensor of data,
/// and optional
///        list of names
/// @param gridBatch The gridbatch to convert
/// @param maybeData Optional JaggedTensor of data to save with the grid batch (one element per
/// voxel)
/// @param maybeNames  Optional list of names for each grid in the batch (or a single name to use
/// for every grid)
/// @return A nanovdb grid handle, whose type is inferred from the data, containing the converted
/// grids
nanovdb::GridHandle<nanovdb::HostBuffer> to_nanovdb(
    const GridBatch                           &gridBatch,
    const std::optional<JaggedTensor>          maybeData  = std::optional<JaggedTensor>(),
    const std::optional<StringOrListOfStrings> maybeNames = std::optional<StringOrListOfStrings>());

/// @brief Save a grid batch and optional jagged tensor to a .nvdb file. Will overwrite existing
/// files.
/// @param path The path to save the file to.
/// @param gridBatch The gridbatch to save
/// @param maybeData Optional JaggedTensor of data to save with the grid batch (one element per
/// voxel)
/// @param maybeNames Optional list of names for each grid in the batch (or a single name to use for
/// every grid)
/// @param compressed Whether to compress the stored grid using Blosc (https://www.blosc.org/)
/// @param verbose Whether to print information about the saved grids
void
save(const std::string &path, const GridBatch &gridBatch,
     const std::optional<JaggedTensor>          maybeData  = std::optional<JaggedTensor>(),
     const std::optional<StringOrListOfStrings> maybeNames = std::optional<StringOrListOfStrings>(),
     bool compressed = false, bool verbose = false);

/// @brief Load a grid batch from a .nvdb file. This function loads each nanovdb grid into the batch
/// as well
///        as a list of tensors containing the data at each grid in the batch
///        (e.g. a Vec3d grid will load a [num_voxels, 3] float64 tensor)
/// @param path The path to the .nvdb file to load
/// @param gridIdentifier The identifier (index, list of indices, name, list of names) to load from
/// the file
/// @param device Which device to load the grid batch on
/// @param verbose If set to true, print information about the loaded grids
/// @return A triple (gridbatch, data, names) where gridbatch is a GridBatch containing the loaded
/// grids,
///         data is a JaggedTensor containing the data of the grids, and names is a list of strings
///         containing the name of each grid
std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
load(const std::string &path, NanoVDBFileGridIdentifier gridIdentifier, const torch::Device &device,
     bool verbose = false);

/// @brief Load a grid batch from a .nvdb file. This function loads each nanovdb grid into the batch
/// as well
///        as a list of tensors containing the data at each grid in the batch
///        (e.g. a Vec3d grid will load a [num_voxels, 3] float64 tensor)
/// @param path The path to the .nvdb file to load
/// @param gridIdentifier The identifier (index, list of indices, name, list of names) to load from
/// the file
/// @param device_string String specifying which device to load the grid batch on
/// @param verbose If set to true, print information about the loaded grids
/// @return A triple (gridbatch, data, names) where gridbatch is a GridBatch containing the loaded
/// grids,
///         data is a JaggedTensor containing the data of the grids, and names is a list of strings
///         containing the name of each grid
std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
load(const std::string &path, NanoVDBFileGridIdentifier gridIdentifier,
     const std::string &device_string, bool verbose = false);

} // namespace fvdb

#endif // FVDB_FVDB_H
