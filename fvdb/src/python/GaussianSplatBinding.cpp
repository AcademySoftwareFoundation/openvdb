// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "TypeCasters.h"
#include <FVDB.h>

#include <torch/extension.h>

#include <GaussianSplatting2.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace pybind11::detail {

template <>
struct type_caster<fvdb::GaussianSplat3d::ProjectionType>
    : public type_caster_base<fvdb::GaussianSplat3d::ProjectionType> {
    using base = type_caster_base<fvdb::GaussianSplat3d::ProjectionType>;

  public:
    fvdb::GaussianSplat3d::ProjectionType projection_type_value;

    bool
    load(handle src, bool convert) {
        std::string strvalue = src.cast<std::string>();
        std::transform(strvalue.begin(), strvalue.end(), strvalue.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (strvalue == "perspective") {
            projection_type_value = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE;
        } else if (strvalue == "orthographic") {
            projection_type_value = fvdb::GaussianSplat3d::ProjectionType::ORTHOGRAPHIC;
        } else {
            return false;
        }
        value = &projection_type_value;
        return true;
    }

    static handle
    cast(const fvdb::GaussianSplat3d::ProjectionType &src, return_value_policy policy,
         handle parent) {
        switch (src) {
        case fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE:
            return pybind11::str("perspective").release();
        case fvdb::GaussianSplat3d::ProjectionType::ORTHOGRAPHIC:
            return pybind11::str("orthographic").release();
        default:
            return pybind11::str("unknown").release();
        }
    }
};

} // namespace pybind11::detail

void
bind_gaussian_splat3d(py::module &m) {
    py::class_<fvdb::GaussianSplat3d::RenderState>(m, "GaussianSplatRenderState")
        .def_property_readonly("means2d", &fvdb::GaussianSplat3d::RenderState::means2d)
        .def_property_readonly("conics", &fvdb::GaussianSplat3d::RenderState::conics)
        .def_property_readonly("render_quantities",
                               &fvdb::GaussianSplat3d::RenderState::renderQuantities)
        .def_property_readonly("depths", &fvdb::GaussianSplat3d::RenderState::depths)
        .def_property_readonly("opacities", &fvdb::GaussianSplat3d::RenderState::opacities)
        .def_property_readonly("radii", &fvdb::GaussianSplat3d::RenderState::radii)
        .def_property_readonly("tile_offsets", &fvdb::GaussianSplat3d::RenderState::offsets)
        .def_property_readonly("tile_gaussian_ids",
                               &fvdb::GaussianSplat3d::RenderState::gaussianIds)
        .def_property_readonly("image_width", &fvdb::GaussianSplat3d::RenderState::imageWidth)
        .def_property_readonly("image_height", &fvdb::GaussianSplat3d::RenderState::imageHeight)
        .def_property_readonly("near_plane", &fvdb::GaussianSplat3d::RenderState::nearPlane)
        .def_property_readonly("far_plane", &fvdb::GaussianSplat3d::RenderState::farPlane)
        .def_property_readonly("projection_type",
                               &fvdb::GaussianSplat3d::RenderState::projectionType)
        .def_property_readonly("sh_degree_to_use",
                               &fvdb::GaussianSplat3d::RenderState::shDegreeToUse)
        .def_property_readonly("min_radius_2d", &fvdb::GaussianSplat3d::RenderState::minRadius2d)
        .def_property_readonly("eps_2d", &fvdb::GaussianSplat3d::RenderState::eps2d)
        .def_property_readonly("antialias", &fvdb::GaussianSplat3d::RenderState::antialias);

    py::class_<fvdb::GaussianSplat3d> gs3d(m, "GaussianSplat3d", "A gaussian splat scene");

    gs3d.def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                      bool>(),
             py::arg("means"), py::arg("quats"), py::arg("scales"), py::arg("opacities"),
             py::arg("sh_coeffs"), py::arg("requires_grad") = false)
        .def_property_readonly("means", &fvdb::GaussianSplat3d::means)
        .def_property_readonly("quats", &fvdb::GaussianSplat3d::quats)
        .def_property_readonly("scales", &fvdb::GaussianSplat3d::scales)
        .def_property_readonly("opacities", &fvdb::GaussianSplat3d::opacities)
        .def_property_readonly("sh_coeffs", &fvdb::GaussianSplat3d::shCoeffs)

        .def("precompute_render_state_for_images",
             &fvdb::GaussianSplat3d::precomputeRenderStateForImages,
             py::arg("world_to_camera_matrices"), py::arg("projection_matrices"),
             py::arg("image_width"), py::arg("image_height"), py::arg("near"), py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1, py::arg("min_radius_2d") = 0.0,
             py::arg("eps_2d") = 0.3, py::arg("antialias") = false)

        .def("precompute_render_state_for_depths",
             &fvdb::GaussianSplat3d::precomputeRenderStateForDepths,
             py::arg("world_to_camera_matrices"), py::arg("projection_matrices"),
             py::arg("image_width"), py::arg("image_height"), py::arg("near"), py::arg("far"),
             py::arg("projection_type") = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("min_radius_2d") = 0.0, py::arg("eps_2d") = 0.3, py::arg("antialias") = false)

        .def("precompute_render_state_for_images_and_depths",
             &fvdb::GaussianSplat3d::precomputeRenderStateForImagesAndDepths,
             py::arg("world_to_camera_matrices"), py::arg("projection_matrices"),
             py::arg("image_width"), py::arg("image_height"), py::arg("near"), py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1, py::arg("min_radius_2d") = 0.0,
             py::arg("eps_2d") = 0.3, py::arg("antialias") = false)

        .def("render_from_state", &fvdb::GaussianSplat3d::renderFromState, py::arg("state"),
             py::arg("crop_width") = -1, py::arg("crop_height") = -1, py::arg("crop_origin_w") = -1,
             py::arg("crop_origin_h") = -1, py::arg("tile_size") = 16)

        .def("render_images", &fvdb::GaussianSplat3d::renderImages,
             py::arg("world_to_camera_matrices"), py::arg("projection_matrices"),
             py::arg("image_width"), py::arg("image_height"), py::arg("near"), py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1, py::arg("tile_size") = 16,
             py::arg("min_radius_2d") = 0.0, py::arg("eps_2d") = 0.3, py::arg("antialias") = false)

        .def("render_depths", &fvdb::GaussianSplat3d::renderDepths,
             py::arg("world_to_camera_matrices"), py::arg("projection_matrices"),
             py::arg("image_width"), py::arg("image_height"), py::arg("near"), py::arg("far"),
             py::arg("projection_type") = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("tile_size") = 16, py::arg("min_radius_2d") = 0.0, py::arg("eps_2d") = 0.3,
             py::arg("antialias") = false)

        .def("render_images_and_depths", &fvdb::GaussianSplat3d::renderImagesAndDepths,
             py::arg("world_to_camera_matrices"), py::arg("projection_matrices"),
             py::arg("image_width"), py::arg("image_height"), py::arg("near"), py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1, py::arg("tile_size") = 16,
             py::arg("min_radius_2d") = 0.0, py::arg("eps_2d") = 0.3, py::arg("antialias") = false);
}
