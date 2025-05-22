// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "TypeCasters.h"

#include <FVDB.h>
#include <GaussianSplatting.h>

#include <torch/extension.h>

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
        std::transform(strvalue.begin(), strvalue.end(), strvalue.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
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
    cast(const fvdb::GaussianSplat3d::ProjectionType &src,
         return_value_policy policy,
         handle parent) {
        switch (src) {
        case fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE:
            return pybind11::str("perspective").release();
        case fvdb::GaussianSplat3d::ProjectionType::ORTHOGRAPHIC:
            return pybind11::str("orthographic").release();
        default: return pybind11::str("unknown").release();
        }
    }
};

} // namespace pybind11::detail

void
bind_gaussian_splat3d(py::module &m) {
    py::class_<fvdb::GaussianSplat3d::ProjectedGaussianSplats>(m, "ProjectedGaussianSplats")
        .def_property_readonly("means2d", &fvdb::GaussianSplat3d::ProjectedGaussianSplats::means2d)
        .def_property_readonly("conics", &fvdb::GaussianSplat3d::ProjectedGaussianSplats::conics)
        .def_property_readonly("render_quantities",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::renderQuantities)
        .def_property_readonly("depths", &fvdb::GaussianSplat3d::ProjectedGaussianSplats::depths)
        .def_property_readonly("opacities",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::opacities)
        .def_property_readonly("radii", &fvdb::GaussianSplat3d::ProjectedGaussianSplats::radii)
        .def_property_readonly("tile_offsets",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::offsets)
        .def_property_readonly("tile_gaussian_ids",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::gaussianIds)
        .def_property_readonly("image_width",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::imageWidth)
        .def_property_readonly("image_height",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::imageHeight)
        .def_property_readonly("near_plane",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::nearPlane)
        .def_property_readonly("far_plane",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::farPlane)
        .def_property_readonly("projection_type",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::projectionType)
        .def_property_readonly("sh_degree_to_use",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::shDegreeToUse)
        .def_property_readonly("min_radius_2d",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::minRadius2d)
        .def_property_readonly("eps_2d", &fvdb::GaussianSplat3d::ProjectedGaussianSplats::eps2d)
        .def_property_readonly("antialias",
                               &fvdb::GaussianSplat3d::ProjectedGaussianSplats::antialias);

    py::class_<fvdb::GaussianSplat3d> gs3d(m, "GaussianSplat3d", "A gaussian splat scene");

    gs3d.def(py::init<torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      bool>(),
             py::arg("means"),
             py::arg("quats"),
             py::arg("log_scales"),
             py::arg("logit_opacities"),
             py::arg("sh0"),
             py::arg("shN"),
             py::arg("requires_grad") = false)
        .def_property("means", &fvdb::GaussianSplat3d::means, &fvdb::GaussianSplat3d::setMeans)
        .def_property("quats", &fvdb::GaussianSplat3d::quats, &fvdb::GaussianSplat3d::setQuats)
        .def_property_readonly("scales", &fvdb::GaussianSplat3d::scales)
        .def_property(
            "log_scales", &fvdb::GaussianSplat3d::logScales, &fvdb::GaussianSplat3d::setLogScales)
        .def_property_readonly("opacities", &fvdb::GaussianSplat3d::opacities)
        .def_property("logit_opacities",
                      &fvdb::GaussianSplat3d::logitOpacities,
                      &fvdb::GaussianSplat3d::setLogitOpacities)
        .def_property("sh0", &fvdb::GaussianSplat3d::sh0, &fvdb::GaussianSplat3d::setSh0)
        .def_property("shN", &fvdb::GaussianSplat3d::shN, &fvdb::GaussianSplat3d::setShN)
        .def_property_readonly("num_gaussians", &fvdb::GaussianSplat3d::numGaussians)
        .def_property_readonly("num_sh_bases", &fvdb::GaussianSplat3d::numShBases)
        .def_property_readonly("num_channels", &fvdb::GaussianSplat3d::numChannels)
        .def_property_readonly("requires_grad", &fvdb::GaussianSplat3d::requiresGrad)
        .def_property("track_max_2d_radii_for_grad",
                      &fvdb::GaussianSplat3d::trackMax2dRadiiForGrad,
                      &fvdb::GaussianSplat3d::setTrackMax2dRadiiForGrad)
        .def_property_readonly("accumulated_mean_2d_gradient_norms_for_grad",
                               &fvdb::GaussianSplat3d::accumulated2dMeansGradientNormsForGrad)
        .def_property_readonly("accumulated_max_2d_radii_for_grad",
                               &fvdb::GaussianSplat3d::accumulatedMax2dRadiiForGrad)
        .def_property_readonly("accumulated_gradient_step_counts_for_grad",
                               &fvdb::GaussianSplat3d::gradientStepCountsForGrad)
        .def_static(
            "from_state_dict",
            [](const std::unordered_map<std::string, torch::Tensor> &stateDict) {
                return fvdb::GaussianSplat3d(stateDict);
            },
            py::arg("state_dict"))
        .def("state_dict", &fvdb::GaussianSplat3d::stateDict)
        .def("load_state_dict", &fvdb::GaussianSplat3d::loadStateDict, py::arg("state_dict"))
        .def_property("requires_grad",
                      &fvdb::GaussianSplat3d::requiresGrad,
                      &fvdb::GaussianSplat3d::setRequiresGrad)
        .def("set_state",
             &fvdb::GaussianSplat3d::setState,
             py::arg("means"),
             py::arg("quats"),
             py::arg("log_scales"),
             py::arg("logit_opacities"),
             py::arg("sh0"),
             py::arg("shN"),
             py::arg("requires_grad") = false)
        .def("save_ply", &fvdb::GaussianSplat3d::savePly, py::arg("filename"))

        .def("reset_grad_state", &fvdb::GaussianSplat3d::resetGradState)
        .def("project_gaussians_for_images",
             &fvdb::GaussianSplat3d::projectGaussiansForImages,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1,
             py::arg("min_radius_2d")    = 0.0,
             py::arg("eps_2d")           = 0.3,
             py::arg("antialias")        = false)

        .def("project_gaussians_for_depths",
             &fvdb::GaussianSplat3d::projectGaussiansForDepths,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type") = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("min_radius_2d")   = 0.0,
             py::arg("eps_2d")          = 0.3,
             py::arg("antialias")       = false)

        .def("project_gaussians_for_images_and_depths",
             &fvdb::GaussianSplat3d::projectGaussiansForImagesAndDepths,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1,
             py::arg("min_radius_2d")    = 0.0,
             py::arg("eps_2d")           = 0.3,
             py::arg("antialias")        = false)

        .def("render_from_projected_gaussians",
             &fvdb::GaussianSplat3d::renderFromProjectedGaussians,
             py::arg("projected_gaussians"),
             py::arg("crop_width")    = -1,
             py::arg("crop_height")   = -1,
             py::arg("crop_origin_w") = -1,
             py::arg("crop_origin_h") = -1,
             py::arg("tile_size")     = 16)

        .def("render_images",
             &fvdb::GaussianSplat3d::renderImages,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1,
             py::arg("tile_size")        = 16,
             py::arg("min_radius_2d")    = 0.0,
             py::arg("eps_2d")           = 0.3,
             py::arg("antialias")        = false)

        .def("render_depths",
             &fvdb::GaussianSplat3d::renderDepths,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type") = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("tile_size")       = 16,
             py::arg("min_radius_2d")   = 0.0,
             py::arg("eps_2d")          = 0.3,
             py::arg("antialias")       = false)

        .def("render_images_and_depths",
             &fvdb::GaussianSplat3d::renderImagesAndDepths,
             py::arg("world_to_camera_matrices"),
             py::arg("projection_matrices"),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("near"),
             py::arg("far"),
             py::arg("projection_type")  = fvdb::GaussianSplat3d::ProjectionType::PERSPECTIVE,
             py::arg("sh_degree_to_use") = -1,
             py::arg("tile_size")        = 16,
             py::arg("min_radius_2d")    = 0.0,
             py::arg("eps_2d")           = 0.3,
             py::arg("antialias")        = false);

    m.def("gaussian_render_jagged",
          &fvdb::gaussianRenderJagged,
          py::arg("means"),
          py::arg("quats"),
          py::arg("scales"),
          py::arg("opacities"),
          py::arg("sh_coeffs"),
          py::arg("viewmats"),
          py::arg("Ks"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("near_plane")           = 0.01,
          py::arg("far_plane")            = 1e10,
          py::arg("sh_degree_to_use")     = 3,
          py::arg("tile_size")            = 16,
          py::arg("radius_clip")          = 0.0,
          py::arg("eps2d")                = 0.3,
          py::arg("antialias")            = false,
          py::arg("render_depth_channel") = false,
          py::arg("return_debug_info")    = false,
          py::arg("return_debug_info")    = false,
          py::arg("ortho")                = false);
}
