
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <vector_types.h>
#include <vector_functions.h>
#include <optix_device.h>

#include "NanoVDB_optix.h"
#include "helpers.h"

#include "RenderFogVolumeUtils.h"
#include "RenderUtils.h"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>

extern "C" {
__constant__ Params constantParams;
}

// -----------------------------------------------------------------------------
extern "C" __global__ void __miss__levelset_radiance()
{
    const MissData* sbt_data = (MissData*)optixGetSbtDataPointer();

    const auto& sceneParams = constantParams.sceneConstants;

    const float3 wRayEye = optixGetWorldRayOrigin();
    const float3 wRayDir = optixGetWorldRayDirection();

    using Vec3T = nanovdb::Vec3f;
    using RayT = nanovdb::Ray<float>;

    float bgIntensity = 0.0f;

    if (sceneParams.useBackground) {
        float groundIntensity = 0.0f;
        float groundMix = 0.0f;

        if (sceneParams.useGround) {
            // intersect with ground plane and draw checker if camera is above...

            float wGroundT = (sceneParams.groundHeight - wRayEye.y) / wRayDir.y;

            if (wRayDir.y != 0 && wGroundT > 0.f) {
                auto wGroundPos = wRayEye + wGroundT * wRayDir;

                groundIntensity = render::evalGroundMaterial(wGroundT, sceneParams.groundFalloff, reinterpret_cast<const Vec3T&>(wGroundPos), wRayDir.y, groundMix);

                if (sceneParams.useLighting && sceneParams.useShadows) {
                    const float3 wLightDir = make_float3(sceneParams.sunDirection[0], sceneParams.sunDirection[1], sceneParams.sunDirection[2]);
                    float        attenuation = 0.0f;

                    optixTrace(
                        constantParams.handle,
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

        float skyIntensity = render::evalSkyMaterial(nanovdb::Vec3f(wRayDir.x, wRayDir.y, wRayDir.z));
        bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
    }

    optixSetPayload_0(float_as_int(bgIntensity));
    optixSetPayload_1(float_as_int(bgIntensity));
    optixSetPayload_2(float_as_int(bgIntensity));
}

// -----------------------------------------------------------------------------
extern "C" __global__ void __miss__env_radiance()
{
    using Vec3T = nanovdb::Vec3f;
    using RayT = nanovdb::Ray<float>;

    const MissData* sbt_data = (MissData*)optixGetSbtDataPointer();
    const float3 wRayEye = optixGetWorldRayOrigin();
    const float3 wRayDir = optixGetWorldRayDirection();

    const auto& sceneParams = constantParams.sceneConstants;

    float bgIntensity = 0.0f;
    if (sceneParams.useBackground) {
        float groundIntensity = 0.0f;
        float groundMix = 0.0f;
        if (sceneParams.useGround) {
            // intersect with ground plane and draw checker if camera is above...
            float wGroundT = (sceneParams.groundHeight - wRayEye.y) / wRayDir.y;
            if (wRayDir.y != 0 && wGroundT > 0.f) {
                auto wGroundPos = wRayEye + wGroundT * wRayDir;
                groundIntensity = render::evalGroundMaterial(wGroundT, sceneParams.groundFalloff, reinterpret_cast<const Vec3T&>(wGroundPos), wRayDir.y, groundMix);
            }
        }
        float skyIntensity = render::evalSkyMaterial(nanovdb::Vec3f(wRayDir.x, wRayDir.y, wRayDir.z));
        bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
    }

    optixSetPayload_0(float_as_int(bgIntensity));
    optixSetPayload_1(float_as_int(bgIntensity));
    optixSetPayload_2(float_as_int(bgIntensity));
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

    const auto& sceneParams = constantParams.sceneConstants;
    const auto& params = constantParams.materialConstants;

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

    const float3 wLightDir = {sceneParams.sunDirection[0], sceneParams.sunDirection[1], sceneParams.sunDirection[2]};
    float        intensity = 1.0f;
    float        occlusion = 0.0f;
    float        voxelUniformSize = float(grid->voxelSize()[0]);

    if (params.useOcclusion > 0) {
        float attenuation = 0.0f;
        auto  pixelSeed = render::hash(float_as_int(wRayDir.x), float_as_int(wRayDir.y));
        float randVar1 = render::randomf(pixelSeed + 0);
        float randVar2 = render::randomf(pixelSeed + 1);
        auto  occDir = render::lambertNoTangent(reinterpret_cast<const Vec3T&>(wNormal), randVar1, randVar2);

        optixTrace(
            constantParams.handle,
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
            occlusion = 1;

        intensity = 1.0f - occlusion;
    }

    if (sceneParams.useLighting) {
        float ambient = 1.0f;
        float shadowFactor = 0.0f;

        float3      wH = normalize(wLightDir - wRayDir);
        float       shadowKey = powf(fmaxf(0.0f, dot(wNormal, wH)), 10.0f);
        const float diffuseWrap = 0.25f;
        float       diffuseKey = fmaxf(0.0f, (dot(wNormal, wLightDir) + diffuseWrap) / (1.0f + diffuseWrap));
        float       diffuseFill = fmaxf(0.0f, -dot(wNormal, wRayDir));

        if (sceneParams.useShadows) {
            float attenuation = 0.0f;

            optixTrace(
                constantParams.handle,
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
                shadowFactor = 1;
        }

        intensity = ((1.0f - shadowFactor) * ((shadowKey * 0.2f) + (diffuseKey * 0.8f)) + (1.0f - occlusion) * (diffuseFill * 0.2f + (ambient * 0.1f)));
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

    const auto& sceneParams = constantParams.sceneConstants;
    const auto& params = constantParams.materialConstants;

    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();

    const auto* densityGrid = reinterpret_cast<const nanovdb::FloatGrid*>(sbt_data->geometry.volume.grid);

    const Vec3T wLightDir = sceneParams.sunDirection;
    const Vec3T iLightDir = densityGrid->worldToIndexDirF(wLightDir).normalize();

    const auto& densityTree = densityGrid->tree();
    const auto  densityAcc = densityTree.getAccessor();
    const auto  densitySampler = nanovdb::createSampler<0, decltype(densityAcc), false>(densityAcc);

    HeterogenousMedium medium;
    medium.densityScale = params.volumeDensityScale;
    medium.densityMin = render::valueToScalar(densityGrid->tree().root().valueMin()) * medium.densityScale;
    medium.densityMax = render::valueToScalar(densityGrid->tree().root().valueMax()) * medium.densityScale;
    medium.densityMax = fmaxf(medium.densityMin, fmaxf(medium.densityMax, 0.001f));
    medium.hgMeanCosine = params.phase;
    medium.temperatureScale = params.volumeTemperatureScale;
    medium.transmittanceMethod = params.transmittanceMethod;
    medium.transmittanceThreshold = params.transmittanceThreshold;
    medium.maxPathDepth = params.maxPathDepth;
    medium.albedo = params.volumeAlbedo;

    const float3 wRayEye = optixGetWorldRayOrigin();
    const float3 wRayDir = optixGetWorldRayDirection();

    const float t0 = optixGetRayTmax();
    const float t1 = int_as_float(optixGetAttribute_0());

    const RayT wRay = RayT(reinterpret_cast<const Vec3T&>(wRayEye),
                           reinterpret_cast<const Vec3T&>(wRayDir),
                           t0,
                           t1);
    RayT       iRay = wRay.worldToIndexF(*densityGrid);

    const uint3    idx = optixGetLaunchIndex();
    const uint3    dim = optixGetLaunchDimensions();
    const uint32_t offset = constantParams.width * idx.y + idx.x;

    auto pixelSeed = render::hash(offset + constantParams.width * constantParams.height * constantParams.numAccumulations);

    Vec3T radiance = Vec3T(0);
    
    iRay.setTimes();
    float throughput;
    bool  isFullyAbsorbed;
    radiance = traceFogVolume(throughput, isFullyAbsorbed, iRay, densitySampler, medium, pixelSeed, sceneParams, iLightDir);

    float3 sceneRadiance = {0.0f, 0.0f, 0.0f};
    if (throughput > 0 && isFullyAbsorbed == false) {
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

    float3 result = sceneRadiance * throughput + make_float3(radiance);

    optixSetPayload_0(float_as_int(result.x));
    optixSetPayload_1(float_as_int(result.y));
    optixSetPayload_2(float_as_int(result.z));
}

extern "C" __global__ void __closesthit__nanovdb_fogvolume_occlusion()
{
    using namespace render::fogvolume;
    using RayT = nanovdb::Ray<float>;
    using Vec3T = nanovdb::Vec3f;

    const auto& sceneParams = constantParams.sceneConstants;
    const auto& params = constantParams.materialConstants;

    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();

    const auto* densityGrid = reinterpret_cast<const nanovdb::FloatGrid*>(sbt_data->geometry.volume.grid);
    const auto& densityTree = densityGrid->tree();
    const auto  densityAcc = densityTree.getAccessor();
    const auto  densitySampler = nanovdb::createSampler<0, decltype(densityAcc), false>(densityAcc);

    HeterogenousMedium medium;
    medium.densityScale = params.volumeDensityScale;
    medium.densityMin = densityGrid->tree().root().valueMin() * medium.densityScale;
    medium.densityMax = densityGrid->tree().root().valueMax() * medium.densityScale;
    medium.densityMax = fmaxf(medium.densityMin, fmaxf(medium.densityMax, 0.001f));
    medium.hgMeanCosine = 0.f;
    medium.albedo = params.volumeAlbedo;

    const float3 wRayEye = optixGetWorldRayOrigin();
    const float3 wRayDir = optixGetWorldRayDirection();

    const float t0 = optixGetRayTmax();
    const float t1 = int_as_float(optixGetAttribute_0());

    const RayT wRay = RayT(reinterpret_cast<const Vec3T&>(wRayEye),
                           reinterpret_cast<const Vec3T&>(wRayDir),
                           t0,
                           t1);
    RayT       iRay = wRay.worldToIndexF(*densityGrid);
    Vec3T      radiance = Vec3T(0);

    const uint3    idx = optixGetLaunchIndex();
    const uint3    dim = optixGetLaunchDimensions();
    const uint32_t offset = constantParams.width * idx.y + idx.x;
    auto           pixelSeed = render::hash(offset + constantParams.width * constantParams.height * constantParams.numAccumulations);

    iRay.setTimes();
    float transmittance = getTransmittance(iRay, densitySampler, medium, pixelSeed);
    optixSetPayload_0(float_as_int(transmittance));
}

extern "C" __global__ void __miss__fogvolume_radiance()
{
    const MissData* sbtData = (MissData*)optixGetSbtDataPointer();

    const auto& sceneParams = constantParams.sceneConstants;

    const float3 wRayEye = optixGetWorldRayOrigin();
    const float3 wRayDir = optixGetWorldRayDirection();

    float groundIntensity = 0.0f;
    float groundMix = 0.0f;

    using Vec3T = nanovdb::Vec3f;
    using RayT = nanovdb::Ray<float>;

    float wT = 1e16f;

    auto radiance = 0.f;

    if (sceneParams.useBackground) {
        if (sceneParams.useGround) {
            // intersect with ground plane and draw checker if camera is above...

            float wGroundT = (sceneParams.groundHeight - wRayEye.y) / wRayDir.y;

            if (wRayDir.y != 0 && wGroundT > 0.f) {
                float3 wGroundPos = wRayEye + wGroundT * wRayDir;

                groundIntensity = render::evalGroundMaterial(wGroundT, sceneParams.groundFalloff, reinterpret_cast<const Vec3T&>(wGroundPos), wRayDir.y, groundMix);

                if (sceneParams.useLighting && sceneParams.useShadows > 0) {
                    const float3 wLightDir = {sceneParams.sunDirection[0], sceneParams.sunDirection[1], sceneParams.sunDirection[2]};

                    // HACK: temporrary hack to ensure the ray is not within the volume.
                    wGroundPos -= 1.0f * wLightDir;

                    float attenuation = 0.0f;
                    optixTrace(
                        constantParams.handle,
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

        float skyIntensity = render::evalSkyMaterial(nanovdb::Vec3f(wRayDir.x, wRayDir.y, wRayDir.z));

        radiance = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
    }

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
