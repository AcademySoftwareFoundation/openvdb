#include <torch/extension.h>

#include "FVDB.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "TypeCasters.h"

void bind_grid_batch(py::module& m);
void bind_jagged_tensor(py::module& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Print types when the user passes in the wrong type
    py::class_<fvdb::Vec3i>(m, "Vec3i");
    py::class_<fvdb::Vec4i>(m, "Vec4i");
    py::class_<fvdb::Vec3d>(m, "Vec3d");
    py::class_<fvdb::Vec3dOrScalar>(m, "Vec3dOrScalar");
    py::class_<fvdb::Vec3iOrScalar>(m, "Vec3iOrScalar");
    py::class_<fvdb::Vec3dBatch>(m, "Vec3dBatch");
    py::class_<fvdb::Vec3dBatchOrScalar>(m, "Vec3dBatchOrScalar");
    py::class_<fvdb::Vec3iBatch>(m, "Vec3iBatch");
    py::class_<fvdb::TorchDeviceOrString>(m, "TorchDeviceOrString");
    py::class_<fvdb::NanoVDBFileGridIdentifier>(m, "NanoVDBFileGridIdentifier");

    bind_grid_batch(m);
    bind_jagged_tensor(m);

    //
    // Utility functions
    //

    // volume rendering
    // TODO: (@fwilliams) JaggedTensor interface
    m.def("volume_render", &fvdb::volumeRender,
          py::arg("sigmas"), py::arg("rgbs"),
          py::arg("deltaTs"), py::arg("ts"),
          py::arg("packInfo"), py::arg("transmittanceThresh"));

    // attention
    m.def("scaled_dot_product_attention", &fvdb::scaledDotProductAttention,
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("scale"), R"_FVDB_(
      Computes scaled dot product attention on query, key and value tensors.
            Different SDP kernels could be chosen similar to
            https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

      Args:
            query (JaggedTensor): A JaggedTensor of shape [B, -1, H, E] of the query.
                  Here B is the batch size, H is the number of heads, E is the embedding size.
            key (JaggedTensor): A JaggedTensor of shape [B, -1, H, E] of the key.
            value (JaggedTensor): A JaggedTensor of shape [B, -1, H, V] of the value.
                  Here V is the value size. Note that the key and value should have the same shape.
            scale (float): The scale factor for the attention.

      Returns:
            out (JaggedTensor): Attention result of shape [B, -1, H, V].)_FVDB_");

    // Concatenate grids or jagged tensors
    m.def("cat", py::overload_cast<const std::vector<fvdb::GridBatch>&>(&fvdb::cat), py::arg("grid_batches"));
    m.def("cat", py::overload_cast<const std::vector<fvdb::JaggedTensor>&, int>(&fvdb::cat), py::arg("jagged_tensors"), py::arg("dim") = 0);

    // Build a jagged tensor from a grid batch or another jagged tensor and a data tensor. They will have the same offset structure
    // m.def("jagged_like", py::overload_cast<fvdb::JaggedTensor, torch::Tensor>(&fvdb::jagged_like), py::arg("like"), py::arg("data"));
    // m.def("jagged_like", py::overload_cast<fvdb::GridBatch, torch::Tensor>(&fvdb::jagged_like), py::arg("like"), py::arg("data"));

    // Static grid construction
    m.def("sparse_grid_from_points", &fvdb::sparse_grid_from_points,
          py::arg("points"),
          py::arg("pad_min") = torch::zeros({3}, torch::kInt32),
          py::arg("pad_max") = torch::zeros({3}, torch::kInt32),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins") = torch::zeros({3}),
          py::arg("mutable") = false);
    m.def("sparse_grid_from_nearest_voxels_to_points", &fvdb::sparse_grid_from_nearest_voxels_to_points,
          py::arg("points"),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins") = torch::zeros({3}),
          py::arg("mutable") = false);
    m.def("sparse_grid_from_ijk", &fvdb::sparse_grid_from_ijk,
          py::arg("ijk"),
          py::arg("pad_min") = torch::zeros({3}, torch::kInt32),
          py::arg("pad_max") = torch::zeros({3}, torch::kInt32),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins") = torch::zeros({3}),
          py::arg("mutable") = false);
    m.def("sparse_grid_from_dense", &fvdb::sparse_grid_from_dense,
          py::arg("num_grids"),
          py::arg("dense_dims"),
          py::arg("ijk_min") = torch::zeros(3, torch::kInt32),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins") = torch::zeros({3}),
          py::arg("mask") = nullptr,
          py::arg("device") = "cpu",
          py::arg("mutable") = false);
    m.def("sparse_grid_from_mesh", &fvdb::sparse_grid_from_mesh,
          py::arg("vertices"),
          py::arg("faces"),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins") = torch::zeros({3}),
          py::arg("mutable") = false);

    // Loading and saving grids
    m.def("load",
          &fvdb::load, py::arg("path"),
          py::arg("grid_id") = py::none(),
          py::arg("device") = "cpu",
          py::arg("verbose") = false);
    m.def("save",
          &fvdb::save, py::arg("path"),
          py::arg("grid_batch"),
          py::arg("data") = py::none(),
          py::arg("names") = py::none(),
          py::arg("compressed") = false,
          py::arg("verbose") = false);

    py::enum_<fvdb::ConvPackBackend>(m, "ConvPackBackend")
        .value("GATHER_SCATTER", fvdb::ConvPackBackend::GATHER_SCATTER)
        .value("IGEMM", fvdb::ConvPackBackend::IGEMM)
        .value("CUTLASS", fvdb::ConvPackBackend::CUTLASS)
        .export_values();

    py::class_<fvdb::SparseConvPackInfo>(m, "SparseConvPackInfo")
        .def(py::init<fvdb::Vec3iOrScalar, fvdb::Vec3iOrScalar, fvdb::GridBatch, torch::optional<fvdb::GridBatch>>(),
             py::arg("kernel_size"), py::arg("stride"), py::arg("source_grid"), py::arg("target_grid"))
        .def_property_readonly("neighborhood_map", &fvdb::SparseConvPackInfo::neighborMap)
        .def_property_readonly("neighborhood_sizes", &fvdb::SparseConvPackInfo::neighborSizes)
        .def_property_readonly("use_me", &fvdb::SparseConvPackInfo::useME)
        .def_property_readonly("out_in_map", &fvdb::SparseConvPackInfo::outInMap)
        .def_property_readonly("reorder_loc", &fvdb::SparseConvPackInfo::reorderLoc)
        .def_property_readonly("sorted_mask", &fvdb::SparseConvPackInfo::sortedMask)
        .def_property_readonly("reduced_sorted_mask", &fvdb::SparseConvPackInfo::reducedSortedMask)
        .def_property_readonly("reorder_out_in_map", &fvdb::SparseConvPackInfo::reoderOutInMap)
        .def_property_readonly("use_tf32", &fvdb::SparseConvPackInfo::useTF32)
        .def_property_readonly("out_in_map_bwd", &fvdb::SparseConvPackInfo::outInMapBwd)
        .def_property_readonly("reorder_loc_bwd", &fvdb::SparseConvPackInfo::reorderLocBwd)
        .def_property_readonly("sorted_mask_bwd_w", &fvdb::SparseConvPackInfo::sortedMaskBwdW)
        .def_property_readonly("sorted_mask_bwd_d", &fvdb::SparseConvPackInfo::sortedMaskBwdD)
        .def_property_readonly("reorder_out_in_map_bwd", &fvdb::SparseConvPackInfo::reorderOutInMapBwd)
        .def_property_readonly("halo_index_buffer", &fvdb::SparseConvPackInfo::haloIndexBuffer)
        .def_property_readonly("output_index_buffer", &fvdb::SparseConvPackInfo::outputIndexBuffer)
        .def_property_readonly("stride",
            [](const fvdb::SparseConvPackInfo& self) {nanovdb::math::Coord stride = self.stride().value();
                                                      return py::make_tuple(stride.x(), stride.y(), stride.z());})
        .def_property_readonly("kernel_size",
            [](const fvdb::SparseConvPackInfo& self) {nanovdb::math::Coord kernel_size = self.kernelSize().value();
                                                      return py::make_tuple(kernel_size.x(), kernel_size.y(), kernel_size.z());})
        .def_property_readonly("source_grid", &fvdb::SparseConvPackInfo::sourceGrid)
        .def_property_readonly("target_grid", &fvdb::SparseConvPackInfo::targetGrid)
        .def("build_gather_scatter", &fvdb::SparseConvPackInfo::buildGatherScatter, py::arg("use_me") = false)
        .def("build_implicit_gemm", &fvdb::SparseConvPackInfo::buildImplicitGEMM,
            py::arg("sorted") = false, py::arg("split_mask_num") = 1, py::arg("training") = false, py::arg("split_mask_num_bwd") = 1, py::arg("use_tf32") = false)
        .def("build_cutlass", &fvdb::SparseConvPackInfo::buildCutlass, py::arg("benchmark") = false)
        .def("sparse_conv_3d", &fvdb::SparseConvPackInfo::sparseConv3d,
	     "Sparse 3d convolution", py::arg("input"), py::arg("weights"), py::arg("backend") = fvdb::ConvPackBackend::GATHER_SCATTER)
        .def("sparse_transpose_conv_3d", &fvdb::SparseConvPackInfo::sparseTransposeConv3d,
	     "Sparse 3d convolution transpose", py::arg("input"), py::arg("weights"), py::arg("backend") = fvdb::ConvPackBackend::GATHER_SCATTER);
}


TORCH_LIBRARY(my_classes, m) {
    m.class_<fvdb::GridBatch>("GridBatch");
    m.class_<fvdb::JaggedTensor>("JaggedTensor");
    m.class_<fvdb::SparseConvPackInfo>("SparseConvPackInfo");
    m.class_<fvdb::detail::GridBatchImpl>("GridBatchImpl");
}
