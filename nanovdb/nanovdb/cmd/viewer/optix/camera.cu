
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <optix_device.h>
#include "NanoVDB_optix.h"
#include "helpers.h"
#include <nanovdb/util/Ray.h>
#include "RenderUtils.h"

extern "C" {
__constant__ Params constantParams;
}

extern "C" __global__ void __raygen__nanovdb_camera()
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    const uint3    idx = optixGetLaunchIndex();
    const uint3    dim = optixGetLaunchDimensions();
    int            ix = idx.x;
    int            iy = idx.y;
    const uint32_t offset = constantParams.width * idx.y + idx.x;
    const auto&    sceneParams = constantParams.sceneConstants;

    float3 color = {0, 0, 0};

    for (int sampleIndex = 0; sampleIndex < sceneParams.samplesPerPixel; ++sampleIndex) {
        uint32_t pixelSeed = render::hash((sampleIndex + (constantParams.numAccumulations + 1) * sceneParams.samplesPerPixel)) ^ render::hash(ix, iy);

        RayT wRay = render::getRayFromPixelCoord(ix, iy, constantParams.width, constantParams.height, constantParams.numAccumulations, sceneParams.samplesPerPixel, pixelSeed, sceneParams);

        float3 result;
        optixTrace(
            constantParams.handle,
            make_float3(wRay.eye()),
            make_float3(wRay.dir()),
            constantParams.sceneEpsilon,
            1e16f,
            0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RAY_TYPE_RADIANCE,
            RAY_TYPE_COUNT,
            RAY_TYPE_RADIANCE,
            float3_as_args(result));

        color += result;
    }

    color /= (float)sceneParams.samplesPerPixel;

    if (constantParams.numAccumulations > 1) {
        float3 prevPixel = make_float3(constantParams.imgBuffer[offset]);

        float3 oldLinearPixel;
        if (sceneParams.useTonemapping)
            render::invTonemapReinhard(*(nanovdb::Vec3f*)&oldLinearPixel, *(nanovdb::Vec3f*)&prevPixel, sceneParams.tonemapWhitePoint);
        else
            render::invTonemapPassthru(*(nanovdb::Vec3f*)&oldLinearPixel, *(nanovdb::Vec3f*)&prevPixel);

        color = oldLinearPixel + (color - oldLinearPixel) * (1.0f / constantParams.numAccumulations);
    }

    if (sceneParams.useTonemapping)
        render::tonemapReinhard(*(nanovdb::Vec3f*)&color, *(nanovdb::Vec3f*)&color, sceneParams.tonemapWhitePoint);

    constantParams.imgBuffer[offset] = make_float4(color, 1.f);
}
