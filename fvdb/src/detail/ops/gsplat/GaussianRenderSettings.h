// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRENDERSETTINGS_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRENDERSETTINGS_H

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {
enum ProjectionType { PERSPECTIVE, ORTHOGRAPHIC };

struct RenderSettings {
    enum class RenderMode {
        RGB   = 0,
        DEPTH = 1,
        RGBD  = 2,
    };

    std::uint32_t  imageWidth;
    std::uint32_t  imageHeight;
    ProjectionType projectionType  = ProjectionType::PERSPECTIVE;
    RenderMode     renderMode      = RenderMode::RGB;
    float          nearPlane       = 0.01;
    float          farPlane        = 1e10;
    std::uint32_t  tileSize        = 16;
    float          radiusClip      = 0.0;
    float          eps2d           = 0.3;
    bool           antialias       = false;
    int            shDegreeToUse   = -1;
    int            numDepthSamples = -1;
};
} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRENDERSETTINGS_H
