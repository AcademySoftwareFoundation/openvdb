
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <optix_device.h>
#include "NanoVDB_optix.h"
#include "helpers.h"
#include <nanovdb/util/Ray.h>
#include "RenderUtils.h"

extern "C" {
__constant__ Params params;
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
    const uint32_t offset = params.width * idx.y + idx.x;
    auto&          camera = *(const Camera<float>*)optixGetSbtDataPointer();

    float3 color = {0, 0, 0};

    for (int sampleIndex = 0; sampleIndex < params.constants.samplesPerPixel; ++sampleIndex) {
        uint32_t pixelSeed = render::hash((sampleIndex + (params.numAccumulations + 1) * params.constants.samplesPerPixel)) ^ render::hash(ix, iy);

        float u = ix + 0.5f;
        float v = iy + 0.5f;

        if (params.numAccumulations > 0 || params.constants.samplesPerPixel > 0) {
#if 1
            float jitterX, jitterY;
            render::cmj(jitterX, jitterY, (sampleIndex + (params.numAccumulations + 1) * params.constants.samplesPerPixel) % 64, 8, 8, pixelSeed);
            u += jitterX - 0.5f;
            v += jitterY - 0.5f;
#else
            float randVar1 = render::randomf(pixelSeed + 0);
            float randVar2 = render::randomf(pixelSeed + 1);
            u += randVar1 - 0.5f;
            v += randVar2 - 0.5f;
#endif
        }

        u /= params.width;
        v /= params.height;

        //if (ix == params.width/2 && iy == params.height/2)
        //    printf("pixel(%d, %d, %d, %d, %f, %f)\n", ix, iy, pixelSeed, sampleIndex, u, v);

        RayT wRay = camera.getRay(u, v);

        float3 result;
        optixTrace(
            params.handle,
            make_float3(wRay.eye()),
            make_float3(wRay.dir()),
            params.sceneEpsilon,
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

    color /= (float)params.constants.samplesPerPixel;

    if (params.numAccumulations > 1) {
        float3 prevPixel = make_float3(params.imgBuffer[offset]);

        float3 oldLinearPixel;
        if (params.constants.useTonemapping)
            render::invTonemapReinhard(*(nanovdb::Vec3f*)&oldLinearPixel, *(nanovdb::Vec3f*)&prevPixel, params.constants.tonemapWhitePoint);
        color = oldLinearPixel + (color - oldLinearPixel) * (1.0f / params.numAccumulations);
    }

    if (params.constants.useTonemapping)
        render::tonemapReinhard(*(nanovdb::Vec3f*)&color, *(nanovdb::Vec3f*)&color, params.constants.tonemapWhitePoint);

    params.imgBuffer[offset] = make_float4(color, 1.f);
}
