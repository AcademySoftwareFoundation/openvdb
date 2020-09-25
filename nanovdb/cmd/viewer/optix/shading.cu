
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <vector_types.h>
#include <vector_functions.h>
#include <optix_device.h>

#include "NanoVDB_optix.h"
#include "helpers.h"
//#include "random.h"

#include "RenderFogVolumeUtils.h"
#include "RenderUtils.h"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>

extern "C" {
__constant__ Params params;
}

// -----------------------------------------------------------------------------
extern "C" __global__ void __miss__levelset_radiance()
{
    const MissData* sbt_data = (MissData*)optixGetSbtDataPointer();

    const float3 wRayEye = optixGetWorldRayOrigin();
    const float3 wRayDir = optixGetWorldRayDirection();

    float groundIntensity = 0.0f;
    float groundMix = 0.0f;

    using Vec3T = nanovdb::Vec3f;
    using RayT = nanovdb::Ray<float>;

    if (params.constants.useGround > 0) {
        // intersect with ground plane and draw checker if camera is above...

        float wGroundT = (params.constants.groundHeight - wRayEye.y) / wRayDir.y;

        if (wGroundT > 0.f) {
            auto wGroundPos = wRayEye + wGroundT * wRayDir;

            render::rayTraceGround(wGroundT, params.constants.groundFalloff, reinterpret_cast<const Vec3T&>(wGroundPos), wRayDir.y, groundIntensity, groundMix);

            if (params.constants.useShadows > 0) {
                const float3 wLightDir = make_float3(0.0f, 1.0f, 0.0f);
                float        attenuation = 0.0f;

                optixTrace(
                    params.handle,
                    reinterpret_cast<const float3&>(wGroundPos),
                    wLightDir,
                    0.01f,
                    1e16f,
                    0.0f,
                    OptixVisibilityMask(1),
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                    RAY_TYPE_OCCLUSION,
                    RAY_TYPE_COUNT,
                    RAY_TYPE_OCCLUSION,
                    reinterpret_cast<uint32_t&>(attenuation));

                groundIntensity *= attenuation;
            }
        }
    }

    float skyIntensity = 0.75f + 0.25f * wRayDir.y;
    auto  radiance = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;

    optixSetPayload_0(float_as_int(radiance));
    optixSetPayload_1(float_as_int(radiance));
    optixSetPayload_2(float_as_int(radiance));
}

extern "C" __global__ void __miss__occlusion()
{
    optixSetPayload_0(float_as_int(1.0f));
}

// -----------------------------------------------------------------------------
// LevelSet rendering method
//
extern "C" __global__ void __closesthit__nanovdb_levelset_radiance()
{
    using RayT = nanovdb::Ray<float>;
    using Vec3T = nanovdb::Vec3f;

    const VolumeGeometry* volume = reinterpret_cast<VolumeGeometry*>(optixGetSbtDataPointer());
    const auto*           grid = reinterpret_cast<const nanovdb::FloatGrid*>(volume->grid);
    auto                  acc = grid->tree().getAccessor();

    const float3   wRayDir = optixGetWorldRayDirection();
    nanovdb::Coord ijk = nanovdb::Coord(static_cast<int>(optixGetAttribute_0()), static_cast<int>(optixGetAttribute_1()), static_cast<int>(optixGetAttribute_2()));
    const float3   wSurfacePos = {int_as_float(optixGetAttribute_3()), int_as_float(optixGetAttribute_4()), int_as_float(optixGetAttribute_5())};

    // sample gradient.
    float v0 = acc.getValue(ijk);
    Vec3T iNormal(-v0);
    ijk[0] += 1;
    iNormal[0] += acc.getValue(ijk);
    ijk[0] -= 1;
    ijk[1] += 1;
    iNormal[1] += acc.getValue(ijk);
    ijk[1] -= 1;
    ijk[2] += 1;
    iNormal[2] += acc.getValue(ijk);
    ijk[2] -= 1;
    auto wNormal = make_float3(grid->indexToWorldDirF(iNormal).normalize());

    const float3 wLightDir = {0.0f, 1.0f, 0.0f};
    float        useLighting = params.constants.useLighting;
    float        intensity = 1.0f;
    float        occlusion = 0.0f;
    float        voxelUniformSize = float(grid->voxelSize()[0]);

    if (params.constants.useOcclusion > 0) {
        float attenuation = 0.0f;
        auto  pixelSeed = render::hash(float_as_int(wRayDir.x), float_as_int(wRayDir.y));
        float randVar1 = render::randomf(pixelSeed + 0);
        float randVar2 = render::randomf(pixelSeed + 1);
        auto  occDir = render::lambertNoTangent(reinterpret_cast<const Vec3T&>(wNormal), randVar1, randVar2);

        optixTrace(
            params.handle,
            wSurfacePos - 0.01f * wRayDir,
            make_float3(occDir),
            1e-3f,
            1e+16f,
            0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            RAY_TYPE_OCCLUSION,
            RAY_TYPE_COUNT,
            RAY_TYPE_OCCLUSION,
            reinterpret_cast<uint32_t&>(attenuation));

        if (attenuation < 1)
            occlusion = params.constants.useOcclusion;

        intensity = 1.0f - occlusion;
    }

    if (useLighting > 0) {
        float ambient = 1.0f;
        float shadowFactor = 0.0f;

        float3      wH = normalize(wLightDir - wRayDir);
        float       shadowKey = powf(fmaxf(0.0f, dot(wNormal, wH)), 10.0f);
        const float diffuseWrap = 0.25f;
        float       diffuseKey = fmaxf(0.0f, (dot(wNormal, wLightDir) + diffuseWrap) / (1.0f + diffuseWrap));
        float       diffuseFill = fmaxf(0.0f, -dot(wNormal, wRayDir));

        if (params.constants.useShadows > 0) {
            float attenuation = 0.0f;

            optixTrace(
                params.handle,
                wSurfacePos - 0.01f * wRayDir,
                wLightDir,
                1e-3f,
                1e+16f,
                0.0f,
                OptixVisibilityMask(1),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                RAY_TYPE_OCCLUSION,
                RAY_TYPE_COUNT,
                RAY_TYPE_OCCLUSION,
                reinterpret_cast<uint32_t&>(attenuation));

            if (attenuation < 1)
                shadowFactor = params.constants.useShadows;
        }

        intensity = useLighting * ((1.0f - shadowFactor) * ((shadowKey * 0.2f) + (diffuseKey * 0.8f)) + (1.0f - occlusion) * (diffuseFill * 0.2f + (ambient * 0.1f)));
        intensity = intensity + ((1.0f - useLighting) * (1.0f - occlusion));
    }

    optixSetPayload_0(float_as_int(intensity));
    optixSetPayload_1(float_as_int(intensity));
    optixSetPayload_2(float_as_int(intensity));
}

// -----------------------------------------------------------------------------
// FogVolume render method
//
extern "C" __global__ void __closesthit__nanovdb_fogvolume_radiance()
{
    using namespace render::fogvolume;
    using RayT = nanovdb::Ray<float>;
    using Vec3T = nanovdb::Vec3f;

    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();

    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(sbt_data->geometry.volume.grid);
    const auto& tree = grid->tree();

    const Vec3T wLightDir = Vec3T(0, 1, 0);
    const Vec3T iLightDir = grid->worldToIndexDirF(wLightDir).normalize();

    auto acc = tree.getAccessor();

    HeterogenousMedium medium;
    medium.densityFactor = params.constants.volumeDensity;
    medium.densityMin = grid->tree().root().valueMin() * medium.densityFactor;
    medium.densityMax = medium.densityFactor; //grid->tree().root().valueMax() * medium.densityFactor;
    medium.densityMax = fmaxf(medium.densityMin, fmaxf(medium.densityMax, 0.001f));
    medium.hgMeanCosine = 0.f;

    const float3 wRayEye = optixGetWorldRayOrigin();
    const float3 wRayDir = optixGetWorldRayDirection();

    const float t0 = optixGetRayTmax();
    const float t1 = int_as_float(optixGetAttribute_0());

    const RayT wRay = RayT(reinterpret_cast<const Vec3T&>(wRayEye),
                           reinterpret_cast<const Vec3T&>(wRayDir),
                           t0,
                           t1);
    RayT       iRay = wRay.worldToIndexF(*grid);

    const uint3    idx = optixGetLaunchIndex();
    const uint3    dim = optixGetLaunchDimensions();
    const uint32_t offset = params.width * idx.y + idx.x;

    auto pixelSeed = render::hash(offset + params.width * params.height * params.numAccumulations);

    Vec3T radiance = Vec3T(0);
    if (params.constants.useLighting) {
        iRay.setTimes();
        radiance = traceVolume(iRay, acc, medium, pixelSeed, iLightDir);
    }

    iRay.setTimes();
    float  transmittance = getTransmittance(iRay, acc, medium, pixelSeed);
    float3 sceneRadiance = {0.0f, 0.0f, 0.0f};
    if (transmittance > 0.01f) {
        optixTrace(0, // only run miss program
                   wRayEye,
                   wRayDir,
                   0.0,
                   1e16f,
                   0.0f,
                   OptixVisibilityMask(1),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
                   RAY_TYPE_RADIANCE,
                   RAY_TYPE_COUNT,
                   RAY_TYPE_RADIANCE,
                   float3_as_args(sceneRadiance));
    }

    float3 result = sceneRadiance * transmittance + make_float3(radiance);

    optixSetPayload_0(float_as_int(result.x));
    optixSetPayload_1(float_as_int(result.y));
    optixSetPayload_2(float_as_int(result.z));
}

extern "C" __global__ void __closesthit__nanovdb_fogvolume_occlusion()
{
    using namespace render::fogvolume;
    using RayT = nanovdb::Ray<float>;
    using Vec3T = nanovdb::Vec3f;

    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();

    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(sbt_data->geometry.volume.grid);
    auto        acc = grid->tree().getAccessor();

    HeterogenousMedium medium;
    medium.densityFactor = params.constants.volumeDensity;
    medium.densityMin = grid->tree().root().valueMin() * medium.densityFactor;
    medium.densityMax = medium.densityFactor; //grid->tree().root().valueMax() * medium.densityFactor;
    medium.densityMax = fmaxf(medium.densityMin, fmaxf(medium.densityMax, 0.001f));
    medium.hgMeanCosine = 0.f;

    const float3 wRayEye = optixGetWorldRayOrigin();
    const float3 wRayDir = optixGetWorldRayDirection();

    const float t0 = optixGetRayTmax();
    const float t1 = int_as_float(optixGetAttribute_0());

    const RayT wRay = RayT(reinterpret_cast<const Vec3T&>(wRayEye),
                           reinterpret_cast<const Vec3T&>(wRayDir),
                           t0,
                           t1);
    RayT       iRay = wRay.worldToIndexF(*grid);
    Vec3T      radiance = Vec3T(0);

    const uint3    idx = optixGetLaunchIndex();
    const uint3    dim = optixGetLaunchDimensions();
    const uint32_t offset = params.width * idx.y + idx.x;
    auto           pixelSeed = render::hash(offset + params.width * params.height * params.numAccumulations);

    iRay.setTimes();
    float transmittance = getTransmittance(iRay, acc, medium, pixelSeed);
    optixSetPayload_0(float_as_int(transmittance));
}

extern "C" __global__ void __miss__fogvolume_radiance()
{
    const MissData* sbtData = (MissData*)optixGetSbtDataPointer();

    const float3 wRayEye = optixGetWorldRayOrigin();
    const float3 wRayDir = optixGetWorldRayDirection();

    float groundIntensity = 0.0f;
    float groundMix = 0.0f;

    using Vec3T = nanovdb::Vec3f;
    using RayT = nanovdb::Ray<float>;

    float wT = 1e16f;

    if (params.constants.useGround > 0) {
        // intersect with ground plane and draw checker if camera is above...

        float wGroundT = (params.constants.groundHeight - wRayEye.y) / wRayDir.y;

        if (wGroundT > 0.f) {
            float3 wGroundPos = wRayEye + wGroundT * wRayDir;

            render::rayTraceGround(wGroundT, params.constants.groundFalloff, reinterpret_cast<const Vec3T&>(wGroundPos), wRayDir.y, groundIntensity, groundMix);

            if (params.constants.useShadows > 0) {
                const float3 wLightDir = {0.0f, 1.0f, 0.0f};

                // HACK: temporrary hack to ensure the ray is not within the volume.
                wGroundPos -= 1.0f * wLightDir;

                float attenuation = 0.0f;
                optixTrace(
                    params.handle,
                    wGroundPos,
                    wLightDir,
                    0.01f,
                    1e16f,
                    0.0f,
                    OptixVisibilityMask(1),
                    OPTIX_RAY_FLAG_NONE,
                    RAY_TYPE_OCCLUSION,
                    RAY_TYPE_COUNT,
                    RAY_TYPE_OCCLUSION,
                    reinterpret_cast<uint32_t&>(attenuation));

                groundIntensity *= attenuation;
            }
        }
    }

    float skyIntensity = 0.75f + 0.25f * wRayDir.y;

    auto radiance = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;

    optixSetPayload_0(float_as_int(radiance));
    optixSetPayload_1(float_as_int(radiance));
    optixSetPayload_2(float_as_int(radiance));
}

// -----------------------------------------------------------------------------
// Grid rendering method
//
extern "C" __global__ void __closesthit__nanovdb_grid_radiance()
{
    float3 normal = make_float3(int_as_float(optixGetAttribute_0()),
                                int_as_float(optixGetAttribute_1()),
                                int_as_float(optixGetAttribute_2()));
    if (normal.x <= 0.0f && normal.y <= 0.0f && normal.z <= 0.0f) { // magenta, yellow, and cyan for backsides
        normal = make_float3(1.0f) + normal;
    }
    optixSetPayload_0(float_as_int(normal.x));
    optixSetPayload_1(float_as_int(normal.y));
    optixSetPayload_2(float_as_int(normal.z));
}
