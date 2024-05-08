#include <torch/extension.h>

#include "FVDB.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "TypeCasters.h"


void bind_grid_batch(py::module& m) {
    py::class_<fvdb::GridBatch>(m, "GridBatch", "A batch of sparse VDB grids.")
        .def(py::init<fvdb::TorchDeviceOrString, bool>(), py::arg("device") = "cpu", py::arg("mutable") = false)

        // Properties
        .def_property_readonly("total_voxels", &fvdb::GridBatch::total_voxels,
            "The total number of voxels indexed by this batch of grids.")
        .def_property_readonly("total_enabled_voxels", &fvdb::GridBatch::total_enabled_voxels,
            "The total number of enabled voxels indexed by this batch of grids.")
        .def_property_readonly("total_bbox", &fvdb::GridBatch::total_bbox, R"_FVDB_(
            A tensor, total_bbox, of shape [2, 3] where total_bbox = `[[bmin_i, bmin_j, bmin_z=k],
              [bmax_i, bmax_j, bmax_k]]` is the bounding box such that `bmin <= ijk < bmax` for all voxels
              ijk in the batch.
        )_FVDB_")
        .def_property_readonly("mutable", &fvdb::GridBatch::is_mutable, "Whether the grid is mutable.")
        .def_property_readonly("device", &fvdb::GridBatch::device, "The device on which this grid is stored.")
        .def_property_readonly("enabled_mask", &fvdb::GridBatch::enabled_mask,
            "A boolean JaggedTensor of shape [B, -1] indicating whether each voxel in the grid is enabled or not.")
        .def_property_readonly("disabled_mask", &fvdb::GridBatch::disabled_mask,
            "A boolean JaggedTensor of shape [B, -1] indicating whether each voxel in the grid is disabled or not.")
        .def_property_readonly("grid_count", &fvdb::GridBatch::grid_count, "The number of grids indexed by this batch.")
        .def_property_readonly("num_voxels", &fvdb::GridBatch::num_voxels,
            "An integer tensor containing the number of voxels per grid indexed by this batch.")
        .def_property_readonly("cum_voxels", &fvdb::GridBatch::cum_voxels, R"_FVDB_(
            An integer tensor containing the cumulative number of voxels indexed by the grids in this batch.
              i.e. `[nvox_0, nvox_0+nvox_1, nvox_0+nvox_1+nvox_2, ...]`
        )_FVDB_")
        .def_property_readonly("num_enabled_voxels", &fvdb::GridBatch::num_enabled_voxels,
            "An integer tensor containing the number of enabled voxels per grid indexed by this batch. If this grid is not mutable, this will be the same as num_voxels.")
        .def_property_readonly("cum_enabled_voxels", &fvdb::GridBatch::cum_enabled_voxels,
            "An integer tensor containing the cumulative number of voxels enabled in each grid in this batch. i.e. `[nvox_0, nvox_0+nvox_1, nvox_0+nvox_1+nvox_2, ...]`")
        .def_property_readonly("origins", [](const fvdb::GridBatch& self) { return self.origins(torch::kFloat32); },
                               "A [num_grids, 3] tensor of world space origins for each grid in this batch.")
        .def_property_readonly("voxel_sizes", [](const fvdb::GridBatch& self) { return self.voxel_sizes(torch::kFloat32); },
                               "A [num_grids, 3] tensor of voxel sizes for each grid in this batch.")
        .def_property_readonly("total_bytes", &fvdb::GridBatch::total_bytes,
                               "The total number of bytes used by this batch of grids.")
        .def_property_readonly("num_bytes", &fvdb::GridBatch::num_bytes,
                               "A [num_grids] tensor of the number of bytes used by each grid in this batch.")
        .def_property_readonly("total_leaf_nodes", &fvdb::GridBatch::total_leaf_nodes,
                               "The total number of leaf nodes used by this batch of grids.")
        .def_property_readonly("num_leaf_nodes", &fvdb::GridBatch::num_leaf_nodes,
                               "A [num_grids] tensor of the number of leaf nodes used by each grid in this batch.")
        .def_property_readonly("jidx", &fvdb::GridBatch::jidx,
                               "A [total_voxels,] tensor of the jagged index of each voxel in this batch.")
        .def_property_readonly("joffsets", &fvdb::GridBatch::joffsets,
                               "A [num_grids+1,] tensor of the jagged offsets of each grid in this batch.")
        .def_property_readonly("ijk", &fvdb::GridBatch::ijk,
                               "A [num_grids, -1, 3] JaggedTensor of the ijk coordinates of each voxel in this batch.")
        .def_property_readonly("ijk_enabled", &fvdb::GridBatch::ijk_enabled,
                               "A [num_grids, -1, 3] JaggedTensor of the ijk coordinates of each enabled voxel in this batch.")
        .def_property_readonly("viz_edge_network", [](const fvdb::GridBatch& self) { return self.viz_edge_network(false); },
                               "A pair of JaggedTensors `(gv, ge)` of shape [num_grids, -1, 3] and [num_grids, -1, 2] where `gv` are the corner positions of each voxel and `ge` are edge indices indexing into `gv`. This property is useful for visualizing the grid.")
        .def_property_readonly("grid_to_world_matrices", [](const fvdb::GridBatch& self) { return self.grid_to_world_matrices(torch::kFloat32); },
                               "A [num_grids, 4, 4] tensor of the grid to world transformation matrices for each grid in this batch.")
        .def_property_readonly("world_to_grid_matrices", [](const fvdb::GridBatch& self) { return self.world_to_grid_matrices(torch::kFloat32); },
                               "A [num_grids, 4, 4] tensor of the world to grid transformation matrices for each grid in this batch.")
        .def_property_readonly("bbox", &fvdb::GridBatch::bbox,
                               "A [num_grids, 2, 3] tensor of the bounding box of each grid in this batch where `bbox[i, 0]` is the minimimum ijk coordinate of the i^th grid, and `bbox[i, 1]` is the maximum ijk coordinate.")
        .def_property_readonly("dual_bbox", &fvdb::GridBatch::dual_bbox,
                               "A [num_grids, 2, 3] tensor of the bounding box of the dual of each grid in this batch where bbox[i, 0] is the minimimum ijk coordinate of the i^th dual grid, and bbox[i, 1] is the maximum ijk coordinate.")
        .def_property_readonly("address", &fvdb::GridBatch::address,
                               "The memory address of the underlying C++ GridBatch object.")

        // Read a property for a single grid in the batch
        .def("voxel_size_at", [](const fvdb::GridBatch& self, uint32_t bi) { return self.voxel_size_at(bi, torch::kFloat32); },
             "Get the voxel size of the bi^th grid in the batch.")
        .def("origin_at", [](const fvdb::GridBatch& self, uint32_t bi) { return self.origin_at(bi, torch::kFloat32); },
             "Get the origin of the bi^th grid in the batch.")
        .def("num_voxels_at", &fvdb::GridBatch::num_voxels_at,
             "Get the number of voxels in the bi^th grid in the batch.")
        .def("cum_voxels_at", &fvdb::GridBatch::cum_voxels_at,
             "Get the cumulative number of voxels in the bi^th grid in the batch. i.e. `nvox_0+nvox_1+...+nvox_i`")
        .def("num_enabled_voxels_at", &fvdb::GridBatch::num_enabled_voxels_at,
             "Get the number of enabled voxels in the bi^th grid in the batch. If this grid isn't mutable, this returns the same value as num_voxels_at.")
        .def("cum_enabled_voxels_at", &fvdb::GridBatch::cum_enabled_voxels_at,
             "Get the cumulative number of enabled voxels in the bi^th grid in the batch. i.e. `nvox_0+nvox_1+...+nvox_i`. If this grid isn't mutable, this returns the same value as cum_voxels_at.")
        .def("bbox_at", &fvdb::GridBatch::bbox_at, R"_FVDB_(
            Get the bounding box (in voxel coordinates) of the bi^th grid in the batch.

            Args:
                bi (int): The index of the grid to get the bounding box of.

            Returns:
                bbox (torch.Tensor): A tensor, bbox, of shape [2, 3] where bbox = [[bmin_i, bmin_j, bmin_z=k],
                  [bmax_i, bmax_j, bmax_k]] is the bi^th bounding box such that bmin <= ijk < bmax for all voxels
                  ijk in the bi^th grid.
        )_FVDB_", py::arg("bi"))
        .def("dual_bbox_at", &fvdb::GridBatch::dual_bbox_at,
             "Get the bounding box (in voxel coordinates) of the dual of the bi^th grid in the batch.")

        // Create a jagged tensor with the same offsets as this grid batch
        .def("jagged_like", &fvdb::GridBatch::jagged_like, py::arg("data"), py::arg("ignore_disabled") = true,
             R"_FVDB_(
            Create a JaggedTensor with the same offsets as this grid batch.

            Args:
                data (torch.Tensor): A tensor of shape `[total_voxels, *]` to be converted to a JaggedTensor.
                ignore_disabled (bool): Whether to ignore disabled voxels when creating the JaggedTensor.

            Returns:
                jagged_data (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, *]` with the same offsets as this grid batch.
            )_FVDB_")

        // Deal with contiguity
        .def("contiguous", &fvdb::GridBatch::contiguous,
             "Return a contiguous copy of this grid batch.")
        .def("is_contiguous", &fvdb::GridBatch::is_contiguous,
             "Whether this grid batch is contiguous.")

        // Array indexing
        .def("__getitem__", [](const fvdb::GridBatch& self, int64_t idx) {
            return self.index(idx);
        }, R"_FVDB_(
            Get the i^th grid in the batch.

            Args:
                idx (int): The index of the grid to get.

            Returns:
                grid (Grid): The i^th grid in the batch.)_FVDB_")
        .def("__getitem__", [](const fvdb::GridBatch& self, pybind11::slice slice) {
            ssize_t start, stop, step, len;
            if (!slice.compute(self.grid_count(), &start, &stop, &step, &len)) {
                TORCH_CHECK_INDEX(false, "Invalid slice ", py::repr(slice).cast<std::string>());
            }
            TORCH_CHECK_INDEX(step != 0, "step cannot be 0");
            return self.index(start, stop, step);
        }, R"_FVDB_(
            Get a slice of grids in the batch.

            Args:
                slice (slice): The slice of grids to get.

            Returns:
                grids (GridBatch): A GridBatch containing the sliced grids.)_FVDB_")
        .def("__getitem__", [](const fvdb::GridBatch& self, std::vector<bool> idx) {
            return self.index(idx);
        }, R"_FVDB_(
            Get a slice of grids in the batch from a boolean mask.

            Args:
                idx (list of bools): A list of bools indicating which grids to get.

            Returns:
                grids (GridBatch): A GridBatch containing the sliced grids.)_FVDB_")
        .def("__getitem__", [](const fvdb::GridBatch& self, std::vector<int64_t> idx) {
            return self.index(idx);
        },
        R"_FVDB_(
            Get a slice of grids in the batch from a list of indices.

            Args:
                idx (list of ints): A list of indices indicating which grids to get.

            Returns:
                grids (GridBatch): A GridBatch containing the sliced grids.)_FVDB_")
        .def("__getitem__", [](const fvdb::GridBatch& self, torch::Tensor idx) {
            return self.index(idx);
        },
        R"_FVDB_(
            Get a slice of grids in the batch from a tensor of indices.

            Args:
                idx (torch.Tensor): A tensor of indices indicating which grids to get.

            Returns:
                grids (GridBatch): A GridBatch containing the sliced grids.)_FVDB_")

        // length
        .def("__len__", &fvdb::GridBatch::grid_count,
             "The number of grids in this batch.")

        // Setting transformation
        .def("set_global_origin", &fvdb::GridBatch::set_global_origin, py::arg("origin"),
             R"_FVDB_(
            Set the origin of all grids in this batch.

            Args:
                origin (list of floats): The new global origin of this batch of grids.)_FVDB_")
        .def("set_global_voxel_size", &fvdb::GridBatch::set_global_voxel_size, py::arg("voxel_size"),
             R"_FVDB_(
            Set the voxel size of all grids in this batch.

            Args:
                voxel_size (list of floats): The new global voxel size of this batch of grids.)_FVDB_")

        // Grid construction
        .def("set_from_mesh", &fvdb::GridBatch::set_from_mesh,
             py::arg("mesh_vertices"),
             py::arg("mesh_faces"),
             py::arg("voxel_sizes") = 1.0,
             py::arg("origins") = torch::zeros(3, torch::kInt32),
              R"_FVDB_(
            Set the voxels in this grid batch to those which intersect a given triangle mesh

            Args:
                mesh_vertices (JaggedTensor): A JaggedTensor of shape [num_grids, -1, 3] of mesh vertex positions.
                mesh_faces (JaggedTensor): A JaggedTensor of shape [num_grids, -1, 3] of integer indexes into `mesh_vertices` specifying the faces of each mesh.
                voxel_sizes (float, list, tensor): Either a float or triple specifyng the voxel size of all the grids in the batch or a tensor of shape [num_grids, 3] specifying the voxel size for each grid.
                origins (float, list, tensor): Either a float or triple specifyng the world space origin of all the grids in the batch or a tensor of shape [num_grids, 3] specifying the world space origin for each grid.)_FVDB_")
        .def("set_from_points", &fvdb::GridBatch::set_from_points,
                py::arg("points"),
                py::arg("pad_min") = torch::zeros(3, torch::kInt32),
                py::arg("pad_max") = torch::zeros(3, torch::kInt32),
                py::arg("voxel_sizes") = 1.0,
                py::arg("origins") = torch::zeros(3, torch::kInt32),
                R"_FVDB_(
            Set the voxels in this grid batch to those which contain a point in a given point cloud (with optional padding)

            Args:
                points (JaggedTensor): A JaggedTensor of shape [num_grids, -1, 3] of point positions.
                pad_min (triple of ints): Index space minimum bound of the padding region.
                pad_max (triple of ints): Index space maximum bound of the padding region.
                mesh_faces (JaggedTensor): A JaggedTensor of shape [num_grids, -1, 3] of integer indexes into `mesh_vertices` specifying the faces of each mesh.
                voxel_sizes (float, list, tensor): Either a float or triple specifyng the voxel size of all the grids in the batch or a tensor of shape [num_grids, 3] specifying the voxel size for each grid.
                origins (float, list, tensor): Either a float or triple specifyng the world space origin of all the grids in the batch or a tensor of shape [num_grids, 3] specifying the world space origin for each grid.)_FVDB_")
        .def("set_from_dense_grid", &fvdb::GridBatch::set_from_dense_grid,
                py::arg("num_grids"),
                py::arg("dense_dims"),
                py::arg("ijk_min") = torch::zeros(3, torch::kInt32),
                py::arg("voxel_sizes") = 1.0,
                py::arg("origins") = torch::zeros(3),
                py::arg("mask") = nullptr,
                R"_FVDB_(
                    Set the voxels in this grid batch to a dense grid with shape [num_grids, width, height, depth], otpionally masking out certain voxels

                    Args:
                        num_grids (int): The number of grids in the batch
                        dense_dims (triple of ints): The dimensions of the dense grid `[width, height, depth]`
                        ijk_min (triple of ints): Index space minimum bound of the dense grid.
                        voxel_sizes (float, list, tensor): Either a float or triple specifyng the voxel size of all the grids in the batch or a tensor of shape [num_grids, 3] specifying the voxel size for each grid.
                        origins (float, list, tensor): Either a float or triple specifyng the world space origin of all the grids in the batch or a tensor of shape [num_grids, 3] specifying the world space origin for each grid.
                        mask (torch.Tensor): A tensor of shape [num_grids, width, height, depth] of booleans indicating which voxels to include/exclude.
                )_FVDB_")
        .def("set_from_ijk", &fvdb::GridBatch::set_from_ijk,
                py::arg("ijk"),
                py::arg("pad_min") = torch::zeros(3, torch::kInt32),
                py::arg("pad_max") = torch::zeros(3, torch::kInt32),
                py::arg("voxel_sizes") = 1.0,
                py::arg("origins") = torch::zeros(3),
                R"_FVDB_(
                    Set the voxels in this grid batch to those specified by a given set of ijk coordinates (with optional padding)

                    Args:
                        ijk (JaggedTensor): A JaggedTensor of shape [num_grids, -1, 3] of ijk coordinates.
                        pad_min (triple of ints): Index space minimum bound of the padding region.
                        pad_max (triple of ints): Index space maximum bound of the padding region.
                        voxel_sizes (float, list, tensor): Either a float or triple specifyng the voxel size of all the grids in the batch or a tensor of shape [num_grids, 3] specifying the voxel size for each grid.
                        origins (float, list, tensor): Either a float or triple specifyng the world space origin of all the grids in the batch or a tensor of shape [num_grids, 3] specifying the world space origin for each grid.
                )_FVDB_")
        .def("set_from_nearest_voxels_to_points", &fvdb::GridBatch::set_from_nearest_voxels_to_points,
                py::arg("points"), py::arg("voxel_sizes") = 1.0, py::arg("origins") = torch::zeros(3),
                R"_FVDB_(
                    Set the voxels in this grid batch to the nearest voxel to each point in a given point cloud

                    Args:
                        points (JaggedTensor): A JaggedTensor of shape [num_grids, -1, 3] of point positions.
                        voxel_sizes (float, list, tensor): Either a float or triple specifyng the voxel size of all the grids in the batch or a tensor of shape [num_grids, 3] specifying the voxel size for each grid.
                        origins (float, list, tensor): Either a float or triple specifyng the world space origin of all the grids in the batch or a tensor of shape [num_grids, 3] specifying the world space origin for each grid.
                )_FVDB_")

        // Interface with dense grids
        .def("read_into_dense", &fvdb::GridBatch::read_into_dense,
             py::arg("sparse_data"),
             py::arg("min_coord") = nullptr,
             py::arg("grid_size") = nullptr,
             R"_FVDB_(
                Read the data in a tensor indexed by this batch of grids into a dense tensor, setting non indexed values to zero.

                Args:
                    sparse_data (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, *]` of values indexed by this grid batch.
                    min_coord (list of ints): Index space minimum bound of the dense grid.
                    grid_size (list of ints): The dimensions of the dense grid to read into `[width, height, depth]`

                Returns:
                    dense_data (torch.Tensor): A tensor of shape `[num_grids, width, height, depth, *]` of values indexed by this grid batch.
             )_FVDB_")
        .def("read_from_dense", &fvdb::GridBatch::read_from_dense,
                py::arg("dense_data"),
                py::arg("dense_origins") = torch::zeros(3, torch::kInt32),
                R"_FVDB_(
                    Read the data in a dense tensor into a JaggedTensor indexed by this batch of grids. Non-indexed values are ignored.

                    Args:
                        dense_data (torch.Tensor): A tensor of shape `[num_grids, width, height, depth, *]` of values to be read from.
                        dense_origins (list of floats): The ijk coordinate corresponding to `dense_data[*, 0, 0, 0]`.

                    Returns:
                        sparse_data (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, *]` of values indexed by this grid batch.
                )_FVDB_")

        .def("fill_to_grid", &fvdb::GridBatch::fill_to_grid,
                py::arg("features"),
                py::arg("other_grid"),
                py::arg("default_value") = 0.0,
                R"_FVDB_(
                    Given a GridBatch and features associated with it, return a JaggedTensor representing features for this batch of grid.
                    Fill any voxels not in the GridBatch with the default value.

                    Args:
                        features (JaggedTensor): A JaggedTensor of shape `[B, -1, *]` containing features associated with other_grid.
                        other_grid (GridBatch): A GridBatch containing the grid to fill from.
                        default_value (float): The value to fill in for voxels not in the GridBatch (default 0.0).

                    Returns:
                        filled_features (JaggedTensor): A JaggedTensor of shape `[B, -1, *]` of features associated with this batch of grids.
                )_FVDB_")

        // Derived grids
        .def("dual_grid", &fvdb::GridBatch::dual_grid, py::arg("exclude_border") = false, R"_FVDB_(
                Return a batch of grids representing the dual of this batch.
                i.e. The centers of the dual grid correspond to the corners of this grid batch. The `[i, j, k]` coordinate of the dual grid corresponds to the bottom/left/back
                corner of the `[i, j, k]` voxel in this grid batch.

                Args:
                    exclude_border (bool): Whether to exclude the border of the grid batch when computing the dual grid

                Returns:
                    dual_grid (GridBatch): A GridBatch representing the dual of this grid batch.
             )_FVDB_")
        .def("coarsened_grid", &fvdb::GridBatch::coarsened_grid, py::arg("coarsening_factor"), R"_FVDB_(
                Return a batch of grids representing the coarsened version of this batch.
                Each voxel `[i, j, k]` in this grid batch maps to voxel `[i / branchFactor, j / branchFactor, k / branchFactor]` in the coarse batch.

                Args:
                    coarsening_factor (int or 3-tuple of ints): How much to coarsen by (i,e, `(2,2,2)` means take every other voxel from start of window).

                Returns:
                    coarsened_grid (GridBatch): A GridBatch representing the coarsened version of this grid batch.
                )_FVDB_")
        .def("subdivided_grid", &fvdb::GridBatch::subdivided_grid, py::arg("subdiv_factor"), py::arg("mask") = nullptr, R"_FVDB_(
                Subdivide the grid batch into a finer grid batch.
                Each voxel [i, j, k] in this grid batch maps to voxels `[i * subdivFactor, j * subdivFactor, k * subdivFactor]` in the fine batch.

                Args:
                    subdiv_factor (int or 3-tuple of ints): How much to subdivide by (i,e, `(2,2,2)` means subdivide each voxel into 2^3 voxels).
                    mask (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 3]` of booleans indicating which voxels to subdivide.

                Returns:
                    subdivided_grid (GridBatch): A GridBatch representing the subdivided version of this grid batch.
                )_FVDB_")
        .def("clipped_grid", &fvdb::GridBatch::clipped_grid, py::arg("ijk_min"), py::arg("ijk_max"), R"_FVDB_(
                Return a batch of grids representing the clipped version of this batch.
                Each voxel `[i, j, k]` in the input batch is included in the output if it lies within `ijk_min` and `ijk_max`.

                Args:
                    ijk_min (list of int triplets): Index space minimum bound of the clip region.
                    ijk_max (list of int triplets): Index space maximum bound of the clip region.

                Returns:
                    clipped_grid (GridBatch): A GridBatch representing the clipped version of this grid batch.
                )_FVDB_")
        .def("conv_grid", &fvdb::GridBatch::conv_grid, py::arg("kernel_size"), py::arg("stride"),
             R"_FVDB_(
                Return a batch of grids representing the convolution of this batch with a given kernel.
                Each voxel `[i, j, k]` in the output batch is the sum of the voxels in the input batch `[i * stride, j * stride, k * stride]` to `[i * stride + kernel_size, j * stride + kernel_size, k * stride + kernel_size]`.

                Args:
                    kernel_size (int or 3-tuple of ints): The size of the kernel to convolve with.
                    stride (int or 3-tuple of ints): The stride to use when convolving.

                Returns:
                    conv_grid (GridBatch): A GridBatch representing the convolution of this grid batch.
             )_FVDB_")

        // Clipping to a bounding box
        .def("clip", &fvdb::GridBatch::clip, R"_FVDB_(
            Return a batch of grids representing the clipped version of this batch of grids and corresponding features.

            Args:
                features (JaggedTensor): A JaggedTensor of shape `[B, -1, *]` containing features associated with this batch of grids.
                ijk_min (list of int triplets): Index space minimum bound of the clip region.
                ijk_max (list of int triplets): Index space maximum bound of the clip region.

            Returns:
                clipped_features (JaggedTensor): a JaggedTensor of shape `[B, -1, *]` of clipped data.
                clipped_grid (GridBatch): the clipped grid batch.
            )_FVDB_",
            py::arg("features"), py::arg("ijk_min"), py::arg("ijk_max"))

        // Upsampling and pooling
        .def("max_pool", &fvdb::GridBatch::max_pool, R"_FVDB_(
            Downsample this batch of grids using maxpooling.

            Args:
                pool_factor (int or 3-tuple of ints): How much to pool by (i,e, `(2,2,2)` means take max over 2x2x2 from start of window).
                data (JaggedTensor): Data at each voxel in this grid to be downsampled (JaggedTensor of shape `[B, -1, *]`).
                stride (int): The stride to use when pooling
                coarse_grid (GridBatch, optional): An optional coarse grid used to specify the output. This is mainly used
                    for memory efficiency so you can chache grids. If you don't pass it in, we'll just create it for you.

            Returns:
                coarse_data (JaggedTensor): a JaggedTensor of shape `[B, -1, *]` of downsampled data.
                coarse_grid (GridBatch): the downsampled grid batch.
            )_FVDB_",
            py::arg("pool_factor"), py::arg("data"), py::arg("stride") = 0, py::arg("coarse_grid") = nullptr)

        .def("avg_pool", &fvdb::GridBatch::avg_pool, R"_FVDB_(
            Downsample this batch of grids using average pooling.

            Args:
                pool_factor (int or 3-tuple of ints): How much to pool by (i,e, `(2,2,2)` means take average over 2x2x2 from start of window).
                data (JaggedTensor): Data at each voxel in this grid to be downsampled (JaggedTensor of shape `[B, -1, *]`).
                stride (int): The stride to use when pooling
                coarse_grid (GridBatch, optional): An optional coarse grid used to specify the output. This is mainly used
                    for memory efficiency so you can chache grids. If you don't pass it in, we'll just create it for you.

            Returns:
                coarse_data (JaggedTensor): a JaggedTensor of shape `[B, -1, *]` of downsampled data.
                coarse_grid (GridBatch): the downsampled grid batch.
        )_FVDB_",
        py::arg("pool_factor"), py::arg("data"), py::arg("stride") = 0, py::arg("coarse_grid") = nullptr)

        .def("subdivide", &fvdb::GridBatch::subdivide,
             py::arg("subdiv_factor"), py::arg("data"), py::arg("mask") = nullptr, py::arg("fine_grid") = nullptr, R"_FVDB_(
                Subdivide the grid batch and associated data tensor into a finer GridBatch and data tensor using nearest neighbor sampling.
                Each voxel [i, j, k] in this grid batch maps to voxels `[i * subdivFactor, j * subdivFactor, k * subdivFactor]` in the fine batch.
                Each data value in the subdividided data tensor inherits its parent value

                Args:
                    subdiv_factor (int or 3-tuple of ints): How much to subdivide by (i,e, `(2,2,2)` means subdivide each voxel into 2^3 voxels).
                    data (JaggedTensor): A JaggedTensor of shape `[B, -1, *]` containing data associated with this batch of grids.
                    mask (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 3]` of booleans indicating which voxels to subdivide.
                    fine_grid (GridBatch): An optional fine grid used to specify the output. This is mainly used
                        for memory efficiency so you can chache grids. If you don't pass it in, we'll just create it for you.

                Returns:
                    fine_data (JaggedTensor): A JaggedTensor of shape `[B, -1, *]` of data associated with the fine grid batch.
                    fine_grid (GridBatch): A GridBatch representing the subdivided version of this grid batch.
                )_FVDB_")

        // Mutating functions
        .def("disable_ijk", &fvdb::GridBatch::disable_ijk, py::arg("ijk"), R"_FVDB_(
            If this is grid is mutable, disable voxels at the specified coordinates, otherwise throw an exception.
            If the ijk values are already disabled or are not represented in this GridBatch, then this function is no-op.

            Args:
                ijk (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 3]` of ijk coordinates to disable.
        )_FVDB_")
        .def("enable_ijk", &fvdb::GridBatch::enable_ijk, py::arg("ijk"), R"_FVDB_(
            If this is grid is mutable, enable voxels at the specified coordinates, otherwise throw an exception.
            If the ijk values are already enabled or are not represented in this GridBatch, then this function is no-op.

            Args:
                ijk (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 3]` of ijk coordinates to disable.
        )_FVDB_")

        // Grid intersects/contains objects
        .def("points_in_active_voxel", &fvdb::GridBatch::points_in_active_voxel,
             py::arg("xyz"), py::arg("ignore_disabled") = false, R"_FVDB_(
            Given a set of points, return a JaggedTensor of booleans indicating which points are in active voxels.

            Args:
                xyz (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 3]` of point positions.
                ignore_disabled (bool): Whether to ignore disabled voxels when computing the output.

            Returns:
                points_in_active_voxel (JaggedTensor): A JaggedTensor of shape `[num_grids, -1]` of booleans indicating which points are in active voxels.
        )_FVDB_")
        .def("coords_in_active_voxel", &fvdb::GridBatch::coords_in_active_voxel,
             py::arg("ijk"), py::arg("ignore_disabled") = false, R"_FVDB_(
            Given a set of ijk coordinates, return a JaggedTensor of booleans indicating which coordinates are active in this gridbatch

            Args:
                ijk (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 3]` of integer ijk coordinates.
                ignore_disabled (bool): Whether to ignore disabled voxels when computing the output.

            Returns:
                coords_in_active_voxel (JaggedTensor): A JaggedTensor of shape `[num_grids, -1]` of booleans indicating which coordinates are in the grid.
        )_FVDB_")
        .def("cubes_intersect_grid", &fvdb::GridBatch::cubes_intersect_grid,
             py::arg("cube_centers"),
             py::arg("cube_min") = 0.0, py::arg("cube_max") = 0.0,
             py::arg("ignore_disabled") = false, R"_FVDB_(
            Given a set of cube centers and extents, return a JaggedTensor of booleans indicating whether cubes intersect active voxels.

            Args:
                cube_centers (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 3]` of cube centers.
                cube_min (float or triple of floats): The minimum extent of each cube (all cubes have the same size).
                cube_max (float or triple of floats): The maximum extent of the cube (all cubes have the same size).
                ignore_disabled (bool): Whether to ignore disabled voxels when computing the output.

            Returns:
                cubes_intersect_grid (JaggedTensor): A JaggedTensor of shape `[num_grids, -1]` of booleans indicating whether cubes intersect active voxels.
        )_FVDB_")
        .def("cubes_in_grid", &fvdb::GridBatch::cubes_in_grid,
             py::arg("cube_centers"),
             py::arg("cube_min") = 0.0, py::arg("cube_max") = 0.0,
             py::arg("ignore_disabled") = false, R"_FVDB_(
            Given a set of cube centers and extents, return a JaggedTensor of booleans indicating whether cubes fully reside in active voxels.

            Args:
                cube_centers (JaggedTensor): A JaggedTensor of shape `[num_grids, -1, 3]` of cube centers.
                cube_min (float or triple of floats): The minimum extent of each cube (all cubes have the same size).
                cube_max (float or triple of floats): The maximum extent of the cube (all cubes have the same size).
                ignore_disabled (bool): Whether to ignore disabled voxels when computing the output.

            Returns:
                cubes_intersect_grid (JaggedTensor): A JaggedTensor of shape `[num_grids, -1]` of booleans indicating whether cubes fully reside in active voxels.
        )_FVDB_")

        // Indexing functions
        .def("ijk_to_index", &fvdb::GridBatch::ijk_to_index, py::arg("ijk"))
        .def("ijk_to_inv_index", &fvdb::GridBatch::ijk_to_inv_index, py::arg("ijk"))
        .def("neighbor_indexes", &fvdb::GridBatch::neighbor_indexes,
                py::arg("ijk"), py::arg("extent"), py::arg("bitshift") = 0)

        // Ray tracing
        .def("voxels_along_rays", &fvdb::GridBatch::voxels_along_rays,
                py::arg("ray_origins"), py::arg("ray_directions"), py::arg("max_voxels"), py::arg("eps") = 0.0)
        .def("segments_along_rays", &fvdb::GridBatch::segments_along_rays,
                py::arg("ray_origins"), py::arg("ray_directions"), py::arg("max_segments"), py::arg("eps") = 0.0, py::arg("ignore_masked") = false)
        .def("uniform_ray_samples", &fvdb::GridBatch::uniform_ray_samples,
                py::arg("ray_origins"), py::arg("ray_directions"),
                py::arg("t_min"), py::arg("t_max"), py::arg("step_size"),
                py::arg("cone_angle") = 0.0, py::arg("include_end_segments") = true)
        .def("ray_implicit_intersection", &fvdb::GridBatch::ray_implicit_intersection,
                py::arg("ray_origins"), py::arg("ray_directions"),
                py::arg("grid_scalars"), py::arg("eps") = 0.0)

        // Sparse grid operations
        .def("splat_trilinear", &fvdb::GridBatch::splat_trilinear,
                py::arg("points"), py::arg("points_data"))
        .def("splat_bezier", &fvdb::GridBatch::splat_bezier,
                py::arg("points"), py::arg("points_data"))
        .def("sample_trilinear", &fvdb::GridBatch::sample_trilinear,
                py::arg("points"), py::arg("voxel_data"))
        .def("sample_trilinear_with_grad", &fvdb::GridBatch::sample_trilinear_with_grad,
                py::arg("points"), py::arg("voxel_data"))
        .def("sample_bezier", &fvdb::GridBatch::sample_bezier,
                py::arg("points"), py::arg("voxel_data"))
        .def("sample_bezier_with_grad", &fvdb::GridBatch::sample_bezier_with_grad,
                py::arg("points"), py::arg("voxel_data"))

        // Marching cubes
        .def("marching_cubes", &fvdb::GridBatch::marching_cubes,
                py::arg("field"), py::arg("level") = 0.0)

        // Convolution
        .def("sparse_conv_halo", &fvdb::GridBatch::sparse_conv_halo, R"_FVDB_(
            Perform in-grid convolution using fast Halo Buffer method. Currently only supports 3x3x3 kernels.

            Args:
                input (JaggedTensor): A JaggedTensor of shape `[B, -1, *]` containing features associated with this batch of grids.
                weight (torch.Tensor): A tensor of shape `[O, I, 3, 3, 3]` containing the convolution kernel.
                variant (int): Which variant of the Halo Buffer method to use. Currently 8 and 64 are supported.

            Returns:
                out (JaggedTensor): a JaggedTensor of shape `[B, -1, *]` of convolved data.
        )_FVDB_",
        py::arg("input"), py::arg("weight"), py::arg("variant") = 8)

        // Coordinate transform
        .def("grid_to_world", &fvdb::GridBatch::grid_to_world, py::arg("ijk"))
        .def("world_to_grid", &fvdb::GridBatch::world_to_grid, py::arg("xyz"))

        // To device
        .def("to", py::overload_cast<fvdb::TorchDeviceOrString>(&fvdb::GridBatch::to, py::const_), py::arg("device"))
        .def("to", py::overload_cast<const torch::Tensor&>(&fvdb::GridBatch::to, py::const_), py::arg("to_tensor"))
        .def("to", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::GridBatch::to, py::const_), py::arg("to_jtensor"))
        .def("to", py::overload_cast<const fvdb::GridBatch&>(&fvdb::GridBatch::to, py::const_), py::arg("to_grid"))

        // .def("clone", &fvdb::GridBatch::clone) // TODO: We totally want this

        .def("sparse_conv_kernel_map", [](fvdb::GridBatch& self, fvdb::Vec3iOrScalar kernelSize, fvdb::Vec3iOrScalar stride,
                                          torch::optional<fvdb::GridBatch> targetGrid) {
            auto ret = fvdb::SparseConvPackInfo(kernelSize, stride, self, targetGrid);
            return std::make_tuple(ret, ret.targetGrid());
        }, py::arg("kernel_size"), py::arg("stride"), py::arg("target_grid") = nullptr)
        .def(py::pickle(
            [](const fvdb::GridBatch& batchHdl) {
                return batchHdl.serialize().to(batchHdl.device());
            },
            [](torch::Tensor t) {
                return fvdb::GridBatch::deserialize(t.cpu()).to(t.device());
            }
        ));
}
