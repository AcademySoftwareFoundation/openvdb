// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "TypeCasters.h"

#include <fvdb/Config.h>
#include <fvdb/FVDB.h>

#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

void bind_grid_batch(py::module &m);
void bind_jagged_tensor(py::module &m);
void bind_gaussian_splat3d(py::module &m);

#define __FVDB__BUILDER_INNER(FUNC_NAME, FUNC_STR, LSHAPE_TYPE)                           \
    m.def(                                                                                \
        FUNC_STR,                                                                         \
        [](const LSHAPE_TYPE &lshape,                                                     \
           std::optional<const std::vector<int64_t>> &rshape,                             \
           std::optional<torch::ScalarType> dtype,                                        \
           std::optional<torch::Device> device,                                           \
           bool requires_grad,                                                            \
           bool pin_memory) {                                                             \
            const torch::Device device_     = device.value_or(torch::kCPU);               \
            const torch::ScalarType dtype_  = dtype.value_or(torch::kFloat32);            \
            const torch::TensorOptions opts = torch::TensorOptions()                      \
                                                  .dtype(dtype_)                          \
                                                  .device(device_)                        \
                                                  .requires_grad(requires_grad)           \
                                                  .pinned_memory(pin_memory);             \
            const std::vector<int64_t> rshape_ = rshape.value_or(std::vector<int64_t>()); \
            return fvdb::FUNC_NAME(lshape, rshape_, opts);                                \
        },                                                                                \
        py::arg("lshape"),                                                                \
        py::arg("rshape")        = std::nullopt,                                          \
        py::arg("dtype")         = std::nullopt,                                          \
        py::arg("device")        = std::nullopt,                                          \
        py::arg("requires_grad") = false,                                                 \
        py::arg("pin_memory")    = false);                                                   \
    m.def(                                                                                \
        FUNC_STR,                                                                         \
        [](const LSHAPE_TYPE &lshape,                                                     \
           std::optional<const std::vector<int64_t>> &rshape,                             \
           std::optional<torch::ScalarType> dtype,                                        \
           std::optional<std::string> device,                                             \
           bool requires_grad,                                                            \
           bool pin_memory) {                                                             \
            torch::Device device_(device.value_or("cpu"));                                \
            if (device_.is_cuda() && !device_.has_index()) {                              \
                device_.set_index(c10::cuda::current_device());                           \
            }                                                                             \
            const torch::ScalarType dtype_  = dtype.value_or(torch::kFloat32);            \
            const torch::TensorOptions opts = torch::TensorOptions()                      \
                                                  .dtype(dtype_)                          \
                                                  .device(device_)                        \
                                                  .requires_grad(requires_grad)           \
                                                  .pinned_memory(pin_memory);             \
            const std::vector<int64_t> rshape_ = rshape.value_or(std::vector<int64_t>()); \
            return fvdb::FUNC_NAME(lshape, rshape_, opts);                                \
        },                                                                                \
        py::arg("lshape"),                                                                \
        py::arg("rshape")        = std::nullopt,                                          \
        py::arg("dtype")         = std::nullopt,                                          \
        py::arg("device")        = std::nullopt,                                          \
        py::arg("requires_grad") = false,                                                 \
        py::arg("pin_memory")    = false);

#define __FVDB__BUILDER(FUNC_NAME, FUNC_STR)                         \
    __FVDB__BUILDER_INNER(FUNC_NAME, FUNC_STR, std::vector<int64_t>) \
    __FVDB__BUILDER_INNER(FUNC_NAME, FUNC_STR, std::vector<std::vector<int64_t>>)
void
bind_jt_build_functions(py::module &m){
    // clang-format off
    __FVDB__BUILDER(jrand, "jrand")
    __FVDB__BUILDER(jrandn, "jrandn")
    __FVDB__BUILDER(jzeros, "jzeros")
    __FVDB__BUILDER(jones, "jones")
    __FVDB__BUILDER(jones, "jempty")
    // clang-format on
}
#undef __FVDB__BUILDER_INNER
#undef __FVDB__BUILDER

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

    bind_grid_batch(m);
    bind_jagged_tensor(m);
    bind_gaussian_splat3d(m);

    //
    // Utility functions
    //

    // volume rendering
    // TODO: (@fwilliams) JaggedTensor interface
    m.def("volume_render",
          &fvdb::volumeRender,
          py::arg("sigmas"),
          py::arg("rgbs"),
          py::arg("deltaTs"),
          py::arg("ts"),
          py::arg("packInfo"),
          py::arg("transmittanceThresh"));

    // attention
    m.def("scaled_dot_product_attention",
          &fvdb::scaledDotProductAttention,
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("scale"),
          R"_FVDB_(
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
    m.def("jcat",
          py::overload_cast<const std::vector<fvdb::GridBatch> &>(&fvdb::jcat),
          py::arg("grid_batches"));
    m.def("jcat",
          py::overload_cast<const std::vector<fvdb::JaggedTensor> &, std::optional<int64_t>>(
              &fvdb::jcat),
          py::arg("jagged_tensors"),
          py::arg("dim") = std::nullopt);

    // Build a jagged tensor from a grid batch or another jagged tensor and a data tensor. They will
    // have the same offset structure m.def("jagged_like", py::overload_cast<fvdb::JaggedTensor,
    // torch::Tensor>(&fvdb::jagged_like), py::arg("like"), py::arg("data")); m.def("jagged_like",
    // py::overload_cast<fvdb::GridBatch, torch::Tensor>(&fvdb::jagged_like), py::arg("like"),
    // py::arg("data"));

    // Static grid construction
    m.def("gridbatch_from_points",
          &fvdb::gridbatch_from_points,
          py::arg("points"),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins")     = torch::zeros({3}));
    m.def("gridbatch_from_nearest_voxels_to_points",
          &fvdb::gridbatch_from_nearest_voxels_to_points,
          py::arg("points"),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins")     = torch::zeros({3}));
    m.def("gridbatch_from_ijk",
          &fvdb::gridbatch_from_ijk,
          py::arg("ijk"),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins")     = torch::zeros({3}));
    m.def("gridbatch_from_dense",
          &fvdb::gridbatch_from_dense,
          py::arg("num_grids"),
          py::arg("dense_dims"),
          py::arg("ijk_min")     = torch::zeros(3, torch::kInt32),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins")     = torch::zeros({3}),
          py::arg("mask")        = nullptr,
          py::arg("device")      = "cpu");

    m.def(
        "gridbatch_from_dense",
        [](const int64_t numGrids,
           const fvdb::Vec3i &denseDims,
           const fvdb::Vec3i &ijkMin,
           const fvdb::Vec3dBatchOrScalar &voxel_sizes,
           const fvdb::Vec3dBatch &origins,
           std::optional<torch::Tensor> mask,
           const std::string &device) {
            return fvdb::gridbatch_from_dense(numGrids,
                                              denseDims,
                                              ijkMin,
                                              voxel_sizes,
                                              origins,
                                              mask,
                                              fvdb::parseDeviceString(device));
        },
        py::arg("num_grids"),
        py::arg("dense_dims"),
        py::arg("ijk_min")     = torch::zeros(3, torch::kInt32),
        py::arg("voxel_sizes") = 1.0,
        py::arg("origins")     = torch::zeros({3}),
        py::arg("mask")        = nullptr,
        py::arg("device")      = "cpu");

    m.def("gridbatch_from_mesh",
          &fvdb::gridbatch_from_mesh,
          py::arg("vertices"),
          py::arg("faces"),
          py::arg("voxel_sizes") = 1.0,
          py::arg("origins")     = torch::zeros({3}));

    // Loading and saving grids
    m.def("load",
          py::overload_cast<const std::string &,
                            const std::vector<uint64_t> &,
                            const torch::Device &,
                            bool>(&fvdb::load),
          py::arg("path"),
          py::arg("indices"),
          py::arg("device")  = torch::kCPU,
          py::arg("verbose") = false);
    m.def(
        "load",
        [](const std::string &path, uint64_t index, const torch::Device &device, bool verbose) {
            std::vector<uint64_t> indices{index};
            return fvdb::load(path, indices, device, verbose);
        },
        py::arg("path"),
        py::arg("index"),
        py::arg("device")  = torch::kCPU,
        py::arg("verbose") = false);
    m.def("load",
          py::overload_cast<const std::string &,
                            const std::vector<std::string> &,
                            const torch::Device &,
                            bool>(&fvdb::load),
          py::arg("path"),
          py::arg("names"),
          py::arg("device")  = torch::kCPU,
          py::arg("verbose") = false);
    m.def(
        "load",
        [](const std::string &path,
           const std::string &name,
           const torch::Device &device,
           bool verbose) {
            std::vector<std::string> names{name};
            return fvdb::load(path, names, device, verbose);
        },
        py::arg("path"),
        py::arg("name"),
        py::arg("device")  = torch::kCPU,
        py::arg("verbose") = false);
    m.def("load",
          py::overload_cast<const std::string &, const torch::Device &, bool>(&fvdb::load),
          py::arg("path"),
          py::arg("device")  = torch::kCPU,
          py::arg("verbose") = false);

    m.def(
        "load",
        [](const std::string &path,
           const std::vector<uint64_t> &indices,
           const std::string &device,
           bool verbose) {
            return fvdb::load(path, indices, fvdb::parseDeviceString(device), verbose);
        },
        py::arg("path"),
        py::arg("indices"),
        py::arg("device")  = "cpu",
        py::arg("verbose") = false);
    m.def(
        "load",
        [](const std::string &path, uint64_t index, const std::string &device, bool verbose) {
            std::vector<uint64_t> indices{index};
            return fvdb::load(path, indices, fvdb::parseDeviceString(device), verbose);
        },
        py::arg("path"),
        py::arg("index"),
        py::arg("device")  = torch::kCPU,
        py::arg("verbose") = false);

    m.def(
        "load",
        [](const std::string &path,
           const std::vector<std::string> &names,
           const std::string &device,
           bool verbose) {
            return fvdb::load(path, names, fvdb::parseDeviceString(device), verbose);
        },
        py::arg("path"),
        py::arg("names"),
        py::arg("device")  = "cpu",
        py::arg("verbose") = false);
    m.def(
        "load",
        [](const std::string &path,
           const std::string &name,
           const std::string &device,
           bool verbose) {
            std::vector<std::string> names{name};
            return fvdb::load(path, names, fvdb::parseDeviceString(device), verbose);
        },
        py::arg("path"),
        py::arg("name"),
        py::arg("device")  = torch::kCPU,
        py::arg("verbose") = false);

    m.def(
        "load",
        [](const std::string &path, const std::string &device, bool verbose) {
            return fvdb::load(path, fvdb::parseDeviceString(device), verbose);
        },
        py::arg("path"),
        py::arg("device")  = "cpu",
        py::arg("verbose") = false);

    m.def("save",
          py::overload_cast<const std::string &,
                            const fvdb::GridBatch &,
                            const std::optional<fvdb::JaggedTensor>,
                            const std::vector<std::string> &,
                            bool,
                            bool>(&fvdb::save),
          py::arg("path"),
          py::arg("grid_batch"),
          py::arg("data")       = py::none(),
          py::arg("names")      = std::vector<std::string>(),
          py::arg("compressed") = false,
          py::arg("verbose")    = false);
    m.def("save",
          py::overload_cast<const std::string &,
                            const fvdb::GridBatch &,
                            const std::optional<fvdb::JaggedTensor>,
                            const std::string &,
                            bool,
                            bool>(&fvdb::save),
          py::arg("path"),
          py::arg("grid_batch"),
          py::arg("data")       = py::none(),
          py::arg("name")       = std::string(),
          py::arg("compressed") = false,
          py::arg("verbose")    = false);

    /*
              py::overload_cast<const std::vector<int64_t>&,
                                std::optional<const std::vector<int64_t>>&,
                                std::optional<torch::ScalarType>,
                                std::optional<torch::Device>,
                                bool, bool>(
    */
    bind_jt_build_functions(m);

    // Global config
    py::class_<fvdb::Config>(m, "config")
        .def_property_static(
            "enable_ultra_sparse_acceleration",
            [](py::object) { return fvdb::Config::global().ultraSparseAccelerationEnabled(); },
            [](py::object, bool enabled) {
                fvdb::Config::global().setUltraSparseAcceleration(enabled);
            })
        .def_property_static(
            "pedantic_error_checking",
            [](py::object) { return fvdb::Config::global().pendanticErrorCheckingEnabled(); },
            [](py::object, bool enabled) {
                fvdb::Config::global().setPendanticErrorChecking(enabled);
            });

    py::enum_<fvdb::ConvPackBackend>(m, "ConvPackBackend")
        .value("GATHER_SCATTER", fvdb::ConvPackBackend::GATHER_SCATTER)
        .value("IGEMM", fvdb::ConvPackBackend::IGEMM)
        .value("CUTLASS", fvdb::ConvPackBackend::CUTLASS)
        .value("LGGS", fvdb::ConvPackBackend::LGGS)
        .export_values();

    py::class_<fvdb::SparseConvPackInfo>(m, "SparseConvPackInfo")
        .def(py::init<fvdb::Vec3iOrScalar,
                      fvdb::Vec3iOrScalar,
                      fvdb::GridBatch,
                      std::optional<fvdb::GridBatch>>(),
             py::arg("kernel_size"),
             py::arg("stride"),
             py::arg("source_grid"),
             py::arg("target_grid"))
        .def_property_readonly("neighborhood_map", &fvdb::SparseConvPackInfo::neighborMap)
        .def_property_readonly("neighborhood_sizes", &fvdb::SparseConvPackInfo::neighborSizes)
        .def_property_readonly("use_me", &fvdb::SparseConvPackInfo::useME)
        .def_property_readonly("out_in_map", &fvdb::SparseConvPackInfo::outInMap)
        .def_property_readonly("reorder_loc", &fvdb::SparseConvPackInfo::reorderLoc)
        .def_property_readonly("sorted_mask", &fvdb::SparseConvPackInfo::sortedMask)
        .def_property_readonly("reduced_sorted_mask", &fvdb::SparseConvPackInfo::reducedSortedMask)
        .def_property_readonly("reorder_out_in_map", &fvdb::SparseConvPackInfo::reoderOutInMap)

        .def_property_readonly("block_kernel_ranges", &fvdb::SparseConvPackInfo::blockKernelRanges)
        .def_property_readonly("block_kernel_in_idx", &fvdb::SparseConvPackInfo::blockKernelInIdx)
        .def_property_readonly("block_kernel_rel_out_idx",
                               &fvdb::SparseConvPackInfo::blockKernelRelOutIdx)

        .def_property_readonly("use_tf32", &fvdb::SparseConvPackInfo::useTF32)
        .def_property_readonly("out_in_map_bwd", &fvdb::SparseConvPackInfo::outInMapBwd)
        .def_property_readonly("reorder_loc_bwd", &fvdb::SparseConvPackInfo::reorderLocBwd)
        .def_property_readonly("sorted_mask_bwd_w", &fvdb::SparseConvPackInfo::sortedMaskBwdW)
        .def_property_readonly("sorted_mask_bwd_d", &fvdb::SparseConvPackInfo::sortedMaskBwdD)
        .def_property_readonly("reorder_out_in_map_bwd",
                               &fvdb::SparseConvPackInfo::reorderOutInMapBwd)
        .def_property_readonly("halo_index_buffer", &fvdb::SparseConvPackInfo::haloIndexBuffer)
        .def_property_readonly("output_index_buffer", &fvdb::SparseConvPackInfo::outputIndexBuffer)
        .def_property_readonly("stride",
                               [](const fvdb::SparseConvPackInfo &self) {
                                   nanovdb::math::Coord stride = self.stride().value();
                                   return py::make_tuple(stride.x(), stride.y(), stride.z());
                               })
        .def_property_readonly("kernel_size",
                               [](const fvdb::SparseConvPackInfo &self) {
                                   nanovdb::math::Coord kernel_size = self.kernelSize().value();
                                   return py::make_tuple(
                                       kernel_size.x(), kernel_size.y(), kernel_size.z());
                               })
        .def_property_readonly("source_grid", &fvdb::SparseConvPackInfo::sourceGrid)
        .def_property_readonly("target_grid", &fvdb::SparseConvPackInfo::targetGrid)
        .def("build_gather_scatter",
             &fvdb::SparseConvPackInfo::buildGatherScatter,
             py::arg("use_me") = false)
        .def("build_implicit_gemm",
             &fvdb::SparseConvPackInfo::buildImplicitGEMM,
             py::arg("sorted")             = false,
             py::arg("split_mask_num")     = 1,
             py::arg("training")           = false,
             py::arg("split_mask_num_bwd") = 1,
             py::arg("use_tf32")           = false)
        .def("build_cutlass", &fvdb::SparseConvPackInfo::buildCutlass, py::arg("benchmark") = false)
        .def("build_lggs", &fvdb::SparseConvPackInfo::buildLGGS)
        .def("sparse_conv_3d",
             &fvdb::SparseConvPackInfo::sparseConv3d,
             "Sparse 3d convolution",
             py::arg("input"),
             py::arg("weights"),
             py::arg("backend") = fvdb::ConvPackBackend::GATHER_SCATTER)
        .def("sparse_transpose_conv_3d",
             &fvdb::SparseConvPackInfo::sparseTransposeConv3d,
             "Sparse 3d convolution transpose",
             py::arg("input"),
             py::arg("weights"),
             py::arg("backend") = fvdb::ConvPackBackend::GATHER_SCATTER)
        .def("to", &fvdb::SparseConvPackInfo::to, py::arg("to_device"))
        .def(
            "to",
            [](const fvdb::SparseConvPackInfo &self, const std::string &to_device) {
                return self.to(fvdb::parseDeviceString(to_device));
            },
            py::arg("to_device"))
        .def("cuda", &fvdb::SparseConvPackInfo::cuda)
        .def("cpu", &fvdb::SparseConvPackInfo::cpu);
}

TORCH_LIBRARY(my_classes, m) {
    m.class_<fvdb::GridBatch>("GridBatch");
    m.class_<fvdb::JaggedTensor>("JaggedTensor");
    m.class_<fvdb::SparseConvPackInfo>("SparseConvPackInfo");
    m.class_<fvdb::detail::GridBatchImpl>("GridBatchImpl");
}
