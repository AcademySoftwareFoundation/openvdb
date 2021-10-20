// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderFunctions.h
	\brief General C++ implementation of the FogVolume rendering code.
*/

#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/SampleFromVoxels.h>
#include "RenderConstants.h"
#include "RenderUtils.h"

namespace render {
namespace fogvolume {

struct HeterogenousMedium
{
    int                       maxPathDepth;
    float                     densityScale;
    float                     densityMin;
    float                     densityMax;
    float                     hgMeanCosine;
    float                     temperatureScale;
    VolumeTransmittanceMethod transmittanceMethod;
    float                     transmittanceThreshold;
    float                     albedo;
};

template<typename VecT, typename SamplerT>
inline __hostdev__ float sampleDensity(const SamplerT& sampler, const VecT& pos, const HeterogenousMedium& medium)
{
    auto densityValue = sampler(pos);
    return valueToScalar(densityValue) * medium.densityScale;
}

template<typename RayT, typename SamplerT>
inline __hostdev__ float deltaTracking(RayT& ray, SamplerT& sampler, const HeterogenousMedium& medium, uint32_t& seed)
{
    const float densityMaxInv = 1.0f / medium.densityMax;
    float       t = ray.t0();
    do {
        t += -logf(randomXorShift(seed)) * densityMaxInv;
    } while (t < ray.t1() && sampleDensity(sampler, ray(t), medium) * densityMaxInv < randomXorShift(seed));
    return t;
}

template<typename RayT, typename SamplerT>
inline __hostdev__ float getTransmittanceRiemannSum(const RayT& ray, const SamplerT& sampler, const HeterogenousMedium& medium, uint32_t& seed)
{
    //const float& kTransmittanceThreshold = medium.transmittanceThreshold;
    static constexpr float kTransmittanceThreshold = 0.001f;
    static constexpr float kTransmittanceReimannSumDeltaStep = 0.5f;

    float transmittance = 1.f;
    float t = ray.t0() - randomXorShift(seed) * kTransmittanceReimannSumDeltaStep;
    for (; t < ray.t1(); t += kTransmittanceReimannSumDeltaStep) {
        float sigmaT = sampleDensity(sampler, ray(t), medium);
        transmittance *= expf(-sigmaT * kTransmittanceReimannSumDeltaStep);
        if (transmittance < kTransmittanceThreshold) {
            return 0.f;
        }
    }
    return transmittance;
}

#if 0
template<typename RayT, typename SamplerT>
inline __hostdev__ float getTransmittanceDDA(const RayT& ray, const SamplerT& sampler, const HeterogenousMedium& medium, uint32_t& seed)
{
    float              transmittance = 1.f;
    nanovdb::DDA<RayT> dda(ray);
    float              t = 0.0f;
    auto               ijk = dda.voxel();
    float              sigmaT = sampleDensity(sampler, ijk, medium);
    while (dda.step()) {
        float dt = dda.time() - t;
        transmittance *= expf(-sigmaT * dt);
        t = dda.time();
        sigmaT = sampleDensity(sampler, dda.voxel(), medium);
        if (transmittance < medium.transmittanceThreshold) {
            return transmittance;
        }
    }
    return transmittance;
}

template<typename RayT, typename SamplerT>
inline __hostdev__ float getTransmittanceHDDA(const RayT& ray, const SamplerT& sampler, const HeterogenousMedium& medium, uint32_t& seed)
{
    float               transmittance = 1.f;
    nanovdb::HDDA<RayT> dda(ray, 1);
    float               t = dda.time();
    while (dda.step()) {
        float dt = dda.time() - t;
        t = dda.time();
        float sigmaT = sampleDensity(sampler, dda.voxel(), medium);
        transmittance *= expf(-sigmaT * dt);
        if (transmittance < medium.transmittanceThreshold) {
            return transmittance;
        }
    }
    return transmittance;
}
#endif

template<typename RayT, typename SamplerT>
inline __hostdev__ float getTransmittanceDeltaTracking(const RayT& ray, const SamplerT& sampler, const HeterogenousMedium& medium, uint32_t& seed)
{
    // delta tracking.
    // faster due to earlier termination, but we need multiple samples to reduce variance.
    const float densityMaxInv = 1.0f / medium.densityMax;
    const int   nSamples = 2;
    float       transmittance = 0.f;
    for (int n = 0; n < nSamples; n++) {
        float t = ray.t0();
        while (true) {
            t -= logf(randomXorShift(seed)) * densityMaxInv;
            if (t >= ray.t1()) {
                transmittance += 1.0f;
                break;
            }
            if (sampleDensity(sampler, ray(t), medium) * densityMaxInv >= randomXorShift(seed))
                break;
        }
    }
    return transmittance / nSamples;
}

template<typename RayT, typename SamplerT>
inline __hostdev__ float getTransmittanceRatioTracking(const RayT& ray, const SamplerT& sampler, const HeterogenousMedium& medium, uint32_t& seed)
{
    //const float& kTransmittanceThreshold = medium.transmittanceThreshold;
    static constexpr float kTransmittanceThreshold = 0.001f;

    // ratio tracking.
    // slower due to no early termination, but better estimation.
    float densityMaxInv = 1.0f / medium.densityMax;
    float transmittance = 1.f;
    float t = ray.t0();
    while (true) {
        t -= logf(randomLCG(seed)) * densityMaxInv;
        if (t >= ray.t1())
            break;
        float sigmaT = sampleDensity(sampler, ray(t), medium);
        transmittance *= 1.0f - sigmaT * densityMaxInv;
        /*
        // Russian roulette.
        const float prob = 1.f - transmittance;
        if (randomf(seed++) < prob)
            return 0.f;
        transmittance /= 1.f - prob;
        */
        if (transmittance < kTransmittanceThreshold)
            return 0.f;
    }
    return transmittance;
}

template<typename RayT, typename SamplerT>
inline __hostdev__ float getTransmittanceResidualRatioTracking(const RayT& ray, const SamplerT& sampler, const HeterogenousMedium& medium, uint32_t& seed)
{
    //const float& kTransmittanceThreshold = medium.transmittanceThreshold;
    static constexpr float kTransmittanceThreshold = 0.001f;

    // residual ratio tracking.
#if 1
    // control is minimum.
    const float controlDensity = medium.densityMin;
    const float residualDensityMax = fmaxf(0.001f, medium.densityMax - controlDensity);
    const float residualDensityMax1 = residualDensityMax;
#elif 0
    // control is maximum.
    const float controlDensity = medium.densityMax;
    const float residualDensityMax1 = medium.densityMin - controlDensity;
    const float residualDensityMax = fmaxf(0.001f, fabsf(medium.densityMin - controlDensity));
#else
    // control is average.
    const float controlDensity = (medium.densityMax + medium.densityMin) * 0.5f;
    const float residualDensityMax1 = medium.densityMax - controlDensity;
    const float residualDensityMax = fmaxf(0.001f, fabsf(residualDensityMax1));
#endif
    const float residualDensityMaxInv = 1.0f / residualDensityMax;
    float       controlTransmittance = expf(-controlDensity * (ray.t1() - ray.t0()));
    float       residualTransmittance = 1.f;
    float       t = ray.t0() - logf(randomXorShift(seed)) * residualDensityMaxInv;
    while (t < ray.t1()) {
        float sigmaT = sampleDensity(sampler, ray(t), medium);

        auto residualDensity = sigmaT - controlDensity;
        residualTransmittance *= 1.0f - residualDensity / residualDensityMax1;
        t -= logf(randomXorShift(seed)) * residualDensityMaxInv;

        if (residualTransmittance * controlTransmittance < kTransmittanceThreshold)
            return 0.f;
    }
    return residualTransmittance * controlTransmittance;
}

template<typename RayT, typename SamplerT>
inline __hostdev__ float getTransmittance(RayT& ray, const SamplerT& sampler, const HeterogenousMedium& medium, uint32_t& seed)
{
    if (!ray.clip(sampler.accessor().root().bbox()))
        return 1.0f;

    switch (medium.transmittanceMethod) {
    case VolumeTransmittanceMethod::kResidualRatioTracking: return getTransmittanceResidualRatioTracking(ray, sampler, medium, seed);
    case VolumeTransmittanceMethod::kRatioTracking: return getTransmittanceRatioTracking(ray, sampler, medium, seed);
    case VolumeTransmittanceMethod::kDeltaTracking: return getTransmittanceDeltaTracking(ray, sampler, medium, seed);
    case VolumeTransmittanceMethod::kRiemannSum: return getTransmittanceRiemannSum(ray, sampler, medium, seed);
    //case VolumeTransmittanceMethod::kHDDA: return getTransmittanceHDDA(ray, sampler, medium, seed);
    //case VolumeTransmittanceMethod::kDDA: return getTransmittanceDDA(ray, sampler, medium, seed);
    default: return 1.f;
    }
}

inline __hostdev__ float henyeyGreenstein(float g, float cosTheta)
{
    if (g == 0) {
        return 1.0f / (3.14159265359f * 4.f);
    } else {
        float denom = nanovdb::Max(0.001f, 1.f + g * g - 2.f * g * cosTheta);
        return (1.0f / (3.14159265359f * 4.f)) * (1.f - g * g) / (denom * sqrtf(denom));
    }
}

inline __hostdev__ nanovdb::Vec3f sampleHG(const nanovdb::Vec3f& dir, float g, float e1, float e2)
{
    if (g == 0) {
        const float phi = (float)(2.0f * 3.14165f) * e1;
        const float cosTheta = 1.0f - 2.0f * e2;
        const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
        return nanovdb::Vec3f(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
    } else {
        const float phi = 2.0f * 3.14159265359f * e2;
        const float s = 2.0f * e1 - 1.0f;
        const float denom = nanovdb::Max(0.001f, (1.0f + g * s));
        const float f = (1.0f - g * g) / denom;
        const float cosTheta = 0.5f * (1.0f / g) * (1.0f + g * g - f * f);
        const float sinTheta = nanovdb::Sqrt(1.0f - cosTheta * cosTheta);
        const auto  phase = nanovdb::Vec3f(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
        // tangent frame.
        const auto tangent = perpStark(dir);
        const auto bitangent = dir.cross(tangent);
        return (phase[0] * tangent + phase[1] * bitangent + phase[2] * dir).normalize();
    }
}

template<typename Vec3T, typename SamplerT>
inline __hostdev__ Vec3T estimateLight(const Vec3T& pos, const Vec3T& /*dir*/, const SamplerT& sampler, const HeterogenousMedium& medium, uint32_t& seed, const Vec3T& lightDir)
{
    const Vec3T lightRadiance = Vec3T(1) * 1.f;
    auto        shadowRay = nanovdb::Ray<float>(pos, lightDir);
    auto        transmittance = getTransmittance(shadowRay, sampler, medium, seed);
    float       pdf = 1.0f; //henyeyGreenstein(medium.hgMeanCosine, dir.dot(lightDir));
    return pdf * lightRadiance * transmittance;
}

template<typename Vec3T, typename SamplerT>
inline __hostdev__ Vec3T estimateBlackbodyEmission(const Vec3T& pos, const SamplerT& temperatureSampler, const HeterogenousMedium& medium, uint32_t& /*seed*/)
{
    const auto value = temperatureSampler(pos);
    return colorFromTemperature(value, medium.temperatureScale);
}

template<typename DensitySamplerT>
inline __hostdev__ nanovdb::Vec3f traceFogVolume(float& throughput, bool& isFullyAbsorbed, nanovdb::Ray<float>& ray, const DensitySamplerT& densitySampler, const HeterogenousMedium& medium, uint32_t& seed, const SceneRenderParameters& sceneParams, const nanovdb::Vec3f& lightDir)
{
    using namespace nanovdb;

    throughput = 1.0f;
    isFullyAbsorbed = false;

    if (!ray.clip(densitySampler.accessor().root().bbox())) {
        return Vec3f(0.0f);
    }

    // Fix the path depth for performance...
    //const int            kMaxPathDepth = medium.maxPathDepth;
    static constexpr int kMaxPathDepth = 4;

    Vec3f radiance = Vec3f(0.f);
    auto  pathRay = ray;
    int   numInteractions = 0;

    while (numInteractions++ < kMaxPathDepth) {
        float s = deltaTracking(pathRay, densitySampler, medium, seed);
        if (s >= pathRay.t1()) {
            break;
        }

        auto pos = pathRay(s);

        if (sceneParams.useLighting) {
            radiance += throughput * estimateLight(pos, pathRay.dir(), densitySampler, medium, seed, lightDir);
        }

        throughput *= medium.albedo;

        // Russian roulette absorption.
        if (throughput < 0.2f) {
            auto r1 = randomXorShift(seed);
            if (r1 > throughput * 5.0f) {
                isFullyAbsorbed = true;
                return Vec3f(0.0f);
            }
            throughput = 0.2f;
        }

        // modify ray using phase function.
        auto r2 = randomXorShift(seed);
        auto r3 = randomXorShift(seed);
        pathRay = Ray<float>(pos, sampleHG(pathRay.dir(), medium.hgMeanCosine, r2, r3));

        if (!pathRay.clip(densitySampler.accessor().root().bbox())) {
            return Vec3f(0.0f);
        }
    }

    return radiance;
}

template<typename DensitySamplerT, typename EmissionSamplerT>
inline __hostdev__ nanovdb::Vec3f traceBlackbodyVolume(float& throughput, bool& isFullyAbsorbed, nanovdb::Ray<float>& ray, const DensitySamplerT& densitySampler, const EmissionSamplerT& emissionSampler, const HeterogenousMedium& medium, uint32_t& seed, const SceneRenderParameters& sceneParams, const nanovdb::Vec3f& lightDir)
{
    using namespace nanovdb;
    throughput = 1.0f;

    int numInteractions = 0;

    if (!ray.clip(densitySampler.accessor().root().bbox())) {
        return Vec3f(0.0f);
    }

    auto& pathRay = ray;

    Vec3f radiance = Vec3f(0.f);
    isFullyAbsorbed = false;

    // Fix the path depth for performance...
    //const int            kMaxPathDepth = medium.maxPathDepth;
    static constexpr int kMaxPathDepth = 4;

    while (numInteractions++ < kMaxPathDepth) {
        float s = deltaTracking(pathRay, densitySampler, medium, seed);
        if (s >= pathRay.t1()) {
            break;
        }

        auto pos = pathRay(s);

        if (sceneParams.useLighting) {
            radiance += throughput * estimateLight(pos, pathRay.dir(), densitySampler, medium, seed, lightDir);
        }

        radiance += throughput * estimateBlackbodyEmission(pos, emissionSampler, medium, seed);

        throughput *= medium.albedo;

        // Russian roulette absorption.
        if (throughput < 0.2f) {
            auto r1 = randomXorShift(seed);
            if (r1 > throughput * 5.0f) {
                isFullyAbsorbed = true;
                return Vec3f(0.0f); // full absorbtion.
            }
            throughput = 0.2f; // unbias.
        }

        // modify ray using phase function.
        auto r2 = randomXorShift(seed);
        auto r3 = randomXorShift(seed);
        pathRay = Ray<float>(pos, sampleHG(pathRay.dir(), medium.hgMeanCosine, r2, r3));

        if (!pathRay.clip(densitySampler.accessor().root().bbox())) {
            return Vec3f(0.0f);
        }
    }

    return radiance;
}

template<typename BuildT, int InterpolationOrder>
struct RenderVolumeRgba32fFn
{
    using ValueT = typename nanovdb::NanoGrid<BuildT>::ValueType;
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int numAccumulations, const nanovdb::BBoxR /*proxy*/, const nanovdb::NanoGrid<BuildT>* densityGrid, const SceneRenderParameters sceneParams, const MaterialParameters params) const
    {
        float* outPixel = &imgBuffer[4 * (ix + width * iy)];
        float  color[4] = {0, 0, 0, 0};

        if (!densityGrid) {
            RayT wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);

            auto envRadiance = traceEnvironment(wRay, sceneParams);

            color[0] = envRadiance;
            color[1] = envRadiance;
            color[2] = envRadiance;

        } else {
            const Vec3T wLightDir = sceneParams.sunDirection;
            const Vec3T iLightDir = densityGrid->worldToIndexDirF(wLightDir).normalize();

            const auto& densityTree = densityGrid->tree();
            const auto  densityAcc = densityTree.getAccessor();
            const auto  densitySampler = nanovdb::createSampler<InterpolationOrder, decltype(densityAcc), false>(densityAcc);

            HeterogenousMedium medium;
            medium.densityScale = params.volumeDensityScale;
            medium.densityMin = valueToScalar(densityGrid->tree().root().minimum()) * medium.densityScale;
            medium.densityMax = valueToScalar(densityGrid->tree().root().maximum()) * medium.densityScale;
            medium.densityMax = fmaxf(medium.densityMin, fmaxf(medium.densityMax, 0.001f));
            medium.hgMeanCosine = params.phase;
            medium.temperatureScale = params.volumeTemperatureScale;
            medium.transmittanceMethod = params.transmittanceMethod;
            medium.transmittanceThreshold = params.transmittanceThreshold;
            medium.maxPathDepth = params.maxPathDepth;
            medium.albedo = params.volumeAlbedo;

            for (int sampleIndex = 0; sampleIndex < sceneParams.samplesPerPixel; ++sampleIndex) {
                uint32_t pixelSeed = hash((sampleIndex + (numAccumulations + 1) * sceneParams.samplesPerPixel)) ^ hash(ix, iy);

                RayT wRay = getRayFromPixelCoord(ix, iy, width, height, numAccumulations, sceneParams.samplesPerPixel, pixelSeed, sceneParams);

                RayT iRay = wRay.worldToIndexF(*densityGrid);

                bool  isFullyAbsorbed = false;
                float pathThroughput = 1.0f;
                Vec3T radiance = traceFogVolume(pathThroughput, isFullyAbsorbed, iRay, densitySampler, medium, pixelSeed, sceneParams, iLightDir);

                if (!isFullyAbsorbed && sceneParams.useBackground && pathThroughput > 0.0f) {
                    float bgIntensity = 0.0f;
                    float groundIntensity = 0.0f;
                    float groundMix = 0.0f;

                    // BUG: this causes trouble.
                    //wRay = iRay.indexToWorldF(*densityGrid);

                    if (sceneParams.useGround) {
                        // intersect with ground plane and draw checker if camera is above...
                        float wGroundT = (sceneParams.groundHeight - wRay.eye()[1]) / wRay.dir()[1];
                        if (wRay.dir()[1] != 0 && wGroundT > 0.f) {
                            Vec3T wGroundPos = wRay(wGroundT);
                            groundIntensity = evalGroundMaterial(wGroundT, sceneParams.groundFalloff, wGroundPos, wRay.dir()[1], groundMix);

                            if (sceneParams.useLighting && sceneParams.useShadows) {
                                Vec3T iGroundPos = densityGrid->worldToIndexF(wGroundPos);
                                RayT  iShadowRay(iGroundPos, iLightDir);
                                float shadowTransmittance = getTransmittance(iShadowRay, densitySampler, medium, pixelSeed);
                                groundIntensity *= shadowTransmittance;
                            }
                        }
                    }

                    float skyIntensity = evalSkyMaterial(wRay.dir());
                    bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
                    radiance += nanovdb::Vec3f(pathThroughput * bgIntensity);
                }

                color[0] += radiance[0];
                color[1] += radiance[1];
                color[2] += radiance[2];
            }

            for (int k = 0; k < 3; ++k)
                color[k] = color[k] / sceneParams.samplesPerPixel;
        }

        compositeFn(outPixel, color, numAccumulations, sceneParams);
    }
};

template<typename BuildT, int InterpolationOrder>
struct RenderBlackBodyVolumeRgba32fFn
{
    using ValueT = typename nanovdb::NanoGrid<BuildT>::ValueType;
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int numAccumulations, const nanovdb::BBoxR /*proxy*/, const nanovdb::NanoGrid<BuildT>* densityGrid, const nanovdb::NanoGrid<BuildT>* temperatureGrid, const SceneRenderParameters sceneParams, const MaterialParameters params) const
    {
        float* outPixel = &imgBuffer[4 * (ix + width * iy)];
        float  color[4] = {0, 0, 0, 0};

        if (!densityGrid || !temperatureGrid) {
            RayT wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);

            auto envRadiance = traceEnvironment(wRay, sceneParams);

            color[0] = envRadiance;
            color[1] = envRadiance;
            color[2] = envRadiance;

        } else {
            const auto& densityTree = densityGrid->tree();
            const auto& temperatureTree = temperatureGrid->tree();
            const auto  densityAcc = densityTree.getAccessor();
            const auto  temperatureAcc = temperatureTree.getAccessor();
            const auto  densitySampler = nanovdb::createSampler<InterpolationOrder, decltype(densityAcc), false>(densityAcc);
            const auto  temperatureSampler = nanovdb::createSampler<InterpolationOrder, decltype(temperatureAcc), false>(temperatureAcc);

            const Vec3T wLightDir = sceneParams.sunDirection;
            const Vec3T iLightDir = densityGrid->worldToIndexDirF(wLightDir).normalize();

            HeterogenousMedium medium;
            medium.densityScale = params.volumeDensityScale;
            medium.densityMin = valueToScalar(densityTree.root().minimum()) * medium.densityScale;
            medium.densityMax = valueToScalar(densityTree.root().maximum()) * medium.densityScale;
            medium.densityMax = fmaxf(medium.densityMin, fmaxf(medium.densityMax, 0.001f));
            medium.hgMeanCosine = params.phase;
            medium.temperatureScale = params.volumeTemperatureScale;
            medium.transmittanceMethod = params.transmittanceMethod;
            medium.transmittanceThreshold = params.transmittanceThreshold;
            medium.maxPathDepth = params.maxPathDepth;
            medium.albedo = params.volumeAlbedo;

            for (int sampleIndex = 0; sampleIndex < sceneParams.samplesPerPixel; ++sampleIndex) {
                uint32_t pixelSeed = hash((sampleIndex + (numAccumulations + 1) * sceneParams.samplesPerPixel)) ^ hash(ix, iy);

                RayT wRay = getRayFromPixelCoord(ix, iy, width, height, numAccumulations, sceneParams.samplesPerPixel, pixelSeed, sceneParams);

                RayT iRay = wRay.worldToIndexF(*densityGrid);

                bool  isFullyAbsorbed = false;
                float pathThroughput = 1.0f;
                Vec3T radiance = traceBlackbodyVolume(pathThroughput, isFullyAbsorbed, iRay, densitySampler, temperatureSampler, medium, pixelSeed, sceneParams, iLightDir);

                if (!isFullyAbsorbed && sceneParams.useBackground && pathThroughput > 0.0f) {
                    float bgIntensity = 0.0f;
                    float groundIntensity = 0.0f;
                    float groundMix = 0.0f;

                    // BUG: this causes trouble.
                    //wRay = iRay.indexToWorldF(*densityGrid);

                    if (sceneParams.useGround) {
                        // intersect with ground plane and draw checker if camera is above...
                        float wGroundT = (sceneParams.groundHeight - wRay.eye()[1]) / wRay.dir()[1];
                        if (wRay.dir()[1] != 0 && wGroundT > 0.f) {
                            Vec3T wGroundPos = wRay(wGroundT);
                            groundIntensity = evalGroundMaterial(wGroundT, sceneParams.groundFalloff, wGroundPos, wRay.dir()[1], groundMix);

                            if (sceneParams.useLighting && sceneParams.useShadows) {
                                Vec3T iGroundPos = densityGrid->worldToIndexF(wGroundPos);
                                RayT  iShadowRay(iGroundPos, iLightDir);
                                float shadowTransmittance = getTransmittance(iShadowRay, densitySampler, medium, pixelSeed);
                                groundIntensity *= shadowTransmittance;
                            }
                        }
                    }

                    float skyIntensity = evalSkyMaterial(wRay.dir());
                    bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
                    radiance += nanovdb::Vec3f(pathThroughput * bgIntensity);
                }

                color[0] += radiance[0];
                color[1] += radiance[1];
                color[2] += radiance[2];
            }

            for (int k = 0; k < 3; ++k)
                color[k] = color[k] / sceneParams.samplesPerPixel;
        }

        compositeFn(outPixel, color, numAccumulations, sceneParams);
    }
};

struct FogVolumeFastRenderFn
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    template<typename BuildT>
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int numAccumulations, const nanovdb::BBoxR /*proxy*/, const nanovdb::NanoGrid<BuildT>* densityGrid, const SceneRenderParameters sceneParams, const MaterialParameters params) const
    {
        //using ValueT = typename nanovdb::NanoGrid<BuildT>::ValueType;

        float* outPixel = &imgBuffer[4 * (ix + width * iy)];
        float  color[4] = {0, 0, 0, 0};

        if (!densityGrid) {
            RayT wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);

            auto envRadiance = traceEnvironment(wRay, sceneParams);

            color[0] = envRadiance;
            color[1] = envRadiance;
            color[2] = envRadiance;

        } else {
            auto&              densityTree = densityGrid->tree();
            HeterogenousMedium medium;
            medium.densityScale = params.volumeDensityScale;
            medium.densityMin = valueToScalar(densityTree.root().minimum()) * medium.densityScale;
            medium.densityMax = valueToScalar(densityTree.root().maximum()) * medium.densityScale;
            medium.densityMax = fmaxf(medium.densityMin, fmaxf(medium.densityMax, 0.001f));
            medium.transmittanceMethod = params.transmittanceMethod;
            medium.transmittanceThreshold = params.transmittanceThreshold;

            uint32_t pixelSeed = hash((numAccumulations + 1)) ^ hash(ix, iy);

            RayT wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);

            RayT iRay = wRay.worldToIndexF(*densityGrid);

            const auto densityAcc = densityTree.getAccessor();
            const auto densitySampler = nanovdb::createSampler<0, decltype(densityAcc), false>(densityAcc);

            const Vec3T wLightDir = sceneParams.sunDirection;
            const Vec3T iLightDir = densityGrid->worldToIndexDirF(wLightDir).normalize();

            float radiance = 0;
            float pathThroughput = getTransmittance(iRay, densitySampler, medium, pixelSeed);

            if (sceneParams.useBackground && pathThroughput > 0.0f) {
                float bgIntensity = 0.0f;
                float groundIntensity = 0.0f;
                float groundMix = 0.0f;
                if (sceneParams.useGround) {
                    // intersect with ground plane and draw checker if camera is above...
                    float wGroundT = (sceneParams.groundHeight - wRay.eye()[1]) / wRay.dir()[1];
                    if (wGroundT > 0.f) {
                        Vec3T wGroundPos = wRay(wGroundT);
                        groundIntensity = evalGroundMaterial(wGroundT, sceneParams.groundFalloff, wGroundPos, wRay.dir()[1], groundMix);

                        if (sceneParams.useLighting && sceneParams.useShadows) {
                            Vec3T iGroundPos = densityGrid->worldToIndexF(wGroundPos);
                            RayT  iShadowRay(iGroundPos, iLightDir);
                            float shadowTransmittance = getTransmittance(iShadowRay, densitySampler, medium, pixelSeed);
                            groundIntensity *= shadowTransmittance;
                        }
                    }
                }

                float skyIntensity = evalSkyMaterial(wRay.dir());
                bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
                radiance += pathThroughput * bgIntensity;
            }

            color[0] += radiance;
            color[1] += radiance;
            color[2] += radiance;
        }

        compositeFn(outPixel, color, numAccumulations, sceneParams);
    }
};
}
} // namespace render::fogvolume
