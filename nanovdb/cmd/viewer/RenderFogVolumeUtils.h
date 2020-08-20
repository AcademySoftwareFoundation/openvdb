// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderFunctions.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief General C++ implementation of the FogVolume rendering code.
*/

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>
#include "RenderConstants.h"
#include "RenderUtils.h"

#define NANOVDB_VIEWER_USE_RATIO_TRACKED_TRANSMISSION

namespace render {
namespace fogvolume {

template<typename ValueT>
inline __hostdev__ ValueT valueToScalar(const ValueT& v)
{
    return v;
}

inline __hostdev__ float valueToScalar(const nanovdb::Vec3f& v)
{
    return (v[0] + v[1] + v[2]) / 3.0f;
}

inline __hostdev__ double valueToScalar(const nanovdb::Vec3d& v)
{
    return (v[0] + v[1] + v[2]) / 3.0;
}

struct HeterogenousMedium
{
    float densityFactor;
    float densityMin;
    float densityMax;
    float hgMeanCosine;
};

template<typename RayT, typename AccT>
inline __hostdev__ float deltaTracking(RayT& ray, AccT& acc, float densityMax, float densityFactor, uint32_t& seed)
{
    if (!ray.clip(acc.root().bbox()))
        return -1;
    float densityMaxInv = 1.0f / densityMax;
    float t = ray.t0();
    do {
        t += -logf(randomf(seed++)) * densityMaxInv;
    } while (t < ray.t1() && valueToScalar(acc.getValue(nanovdb::Coord::Floor(ray(t)))) * densityFactor * densityMaxInv < randomf(seed++));
    return t;
}

template<typename RayT, typename AccT>
inline __hostdev__ float getTransmittance(RayT& ray, AccT& acc, const HeterogenousMedium& medium, uint32_t& seed)
{
    if (!ray.clip(acc.root().bbox()))
        return 1.0f;

#if 0
    float      transmittance = 1.f;
    nanovdb::HDDA<RayT> dda(ray, 1);
    while (dda.step()) {
        auto  densityValue = acc.getValue(dda.voxel());
        float densityScalar = valueToScalar(densityValue) * medium.densityFactor;
        densityScalar = nanovdb::Clamp(densityScalar, medium.densityMin, medium.densityMax); // just in case these are soft extrema

        transmittance *= 1.f - densityScalar;

        if (transmittance < 0.01f)
            return transmittance;
    }
    return transmittance;
#elif 0
    float       transmittance = 1.f;
    const float dt = 0.5f;
    for (float t = ray.t0(); t < ray.t1(); t += dt) {
        auto  densityValue = acc.getValue(nanovdb::Coord::Floor(ray(t)));
        float densityScalar = valueToScalar(densityValue) * medium.densityFactor;
        densityScalar = nanovdb::Clamp(densityScalar, medium.densityMin, medium.densityMax); // just in case these are soft extrema

        transmittance *= 1.f - densityScalar;

        if (transmittance < 0.01f)
            return transmittance;
    }
    return transmittance;
#elif !defined(NANOVDB_VIEWER_USE_RATIO_TRACKED_TRANSMISSION)
    // delta tracking.
    // faster due to earlier termination, but we need multiple samples
    // to reduce variance.
    const float densityMaxInv = 1.0f / medium.densityMax;
    const int   nSamples = 2;
    float       transmittance = 0.f;
    for (int n = 0; n < nSamples; n++) {
        float t = ray.t0();
        while (true) {
            t -= logf(randomf(seed++)) * densityMaxInv;

            if (t >= ray.t1()) {
                transmittance += 1.0f;
                break;
            }

            auto density = acc.getValue(nanovdb::Coord::Floor(ray(t))) * medium.densityFactor;
            density = nanovdb::Clamp(density, medium.densityMin, medium.densityMax); // just in case these are soft extrema

            if (density * densityMaxInv >= randomf(seed++))
                break;
        }
    }
    return transmittance / nSamples;
#elif 0
    // ratio tracking.
    // slower due to no early termination, but better estimation.
    float densityMaxInv = 1.0f / medium.densityMax;
    float transmittance = 1.f;
    float t = ray.t0();
    while (true) {
        t -= logf(randomf(seed++)) * densityMaxInv;
        if (t >= ray.t1())
            break;
        auto density = acc.getValue(nanovdb::Coord::Floor(ray(t))) * medium.densityFactor;
        density = nanovdb::Clamp(density, medium.densityMin, medium.densityMax); // just in case these are soft extrema

        transmittance *= 1.0f - density * densityMaxInv;
    }
    return transmittance;
#elif 1
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
    float controlTransmittance = expf(-controlDensity * (ray.t1() - ray.t0()));
    float residualTransmittance = 1.f;
    float t = ray.t0() - logf(randomf(seed++)) * residualDensityMaxInv;
    while (t < ray.t1()) {
        const auto densityValue = acc.getValue(nanovdb::Coord::Floor(ray(t)));
        float densityScalar = valueToScalar(densityValue) * medium.densityFactor;
        densityScalar = nanovdb::Clamp(densityScalar, medium.densityMin, medium.densityMax); // just in case these are soft extrema

        auto residualDensity = densityScalar - controlDensity;
        residualTransmittance *= 1.0f - residualDensity / residualDensityMax1;
        t -= logf(randomf(seed++)) * residualDensityMaxInv;
    }
    return residualTransmittance * controlTransmittance;
#endif // NANOVDB_VIEWER_USE_RATIO_TRACKED_TRANSMISSION
}

inline __hostdev__ float henyeyGreenstein(float g, float cosTheta)
{
    // phase function pdf.
#if 1
    // isotropic.
    return 3.14159265359f / 4.f;
#else
    float denom = 1.f + g * g - 2.f * g * cosTheta;
    return (3.14159265359f / 4.f) * (1.f - g * g) / (denom * sqrtf(denom));
#endif
}

inline __hostdev__ nanovdb::Vec3f sampleHG(float g, float e1, float e2)
{
    // phase function.
#if 1
    // isotropic
    const float phi = (float)(2.0f * 3.14165f) * e1;
    const float cosTheta = 1.0f - 2.0f * e2;
    const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
#else
    const float phi = (float)(2.0f * 3.14159265359f) * e2;
    const float s = 2.0f * e1 - 1.0f;
    const float denom = nanovdb::Max(0.001f, (1.0f + g * s));
    const float f = (1.0f - g * g) / denom;
    const float cosTheta = 0.5f * (1.0f / g) * (1.0f + g * g - f * f);
    const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
#endif
    return nanovdb::Vec3f(cosf(phi) * cosTheta, sinf(phi) * sinTheta, cosTheta);
}

template<typename Vec3T, typename AccT>
inline __hostdev__ Vec3T estimateLight(const Vec3T& pos, const Vec3T& dir, const AccT& acc, const HeterogenousMedium& medium, uint32_t& seed, const Vec3T& lightDir)
{
    const Vec3T lightRadiance = Vec3T(1) * 1.f;
    auto        shadowRay = nanovdb::Ray<float>(pos, lightDir);
    auto        transmittance = getTransmittance(shadowRay, acc, medium, seed);
    float       pdf = henyeyGreenstein(medium.hgMeanCosine, dir.dot(lightDir));
    return pdf * lightRadiance * transmittance;
}

template<typename RayT, typename AccT>
inline __hostdev__ nanovdb::Vec3f traceVolume(RayT& ray, AccT& acc, const HeterogenousMedium& medium, uint32_t& seed, const nanovdb::Vec3f& lightDir)
{
    using namespace nanovdb;
    using Vec3T = typename RayT::Vec3T;
    float throughput = 1.0f;

    float albedo = 0.8f;

    int max_interactions = 40;
    int num_interactions = 0;

    if (!ray.clip(acc.root().bbox())) {
        return Vec3f(0.0f);
    }

    RayT pathRay = ray;

    Vec3f radiance = Vec3f(0.f);

    while (true) {
        float s = deltaTracking(pathRay, acc, medium.densityMax, medium.densityFactor, seed);
        if (s >= pathRay.t1())
            break;

        if (s < 0)
            return Vec3f(0.0f);

        if (num_interactions++ >= max_interactions)
            return Vec3f(0.0f);

        auto pos = pathRay(s);

        // sample key light.
        radiance = radiance + throughput * estimateLight(pos, pathRay.dir(), acc, medium, seed, lightDir);

        throughput *= albedo;

        // Russian roulette absorption.
        if (throughput < 0.2f) {
            auto r1 = randomf(seed++);
            if (r1 > throughput * 5.0f)
                return Vec3f(0.0f); // full absorbtion.
            throughput = 0.2f; // unbias.
        }

        // modify ray using phase function.
        auto r2 = randomf(seed++);
        auto r3 = randomf(seed++);
        pathRay = RayT(pos, sampleHG(medium.hgMeanCosine, r2, r3));

        if (!pathRay.clip(acc.root().bbox()))
            return Vec3f(0.0f);
    }
    /*
	const float f = (0.5f + 0.5f * ray.dir()[1]) * throughput;
	radiance = radiance + Vec3f(f);*/

    return radiance;
}

struct RenderVolumeRgba32fFn
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    template<typename ValueT>
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, Camera<float> camera, const nanovdb::NanoGrid<ValueT>* grid, int numAccumulations, const RenderConstants params) const
    {
        float* outPixel = &imgBuffer[4 * (ix + width * iy)];

        const auto& tree = grid->tree();

        const Vec3T wLightDir = Vec3T(0, 1, 0);
        const Vec3T iLightDir = grid->worldToIndexDirF(wLightDir).normalize();

        auto acc = tree.getAccessor();

        HeterogenousMedium medium;
        medium.densityFactor = params.volumeDensity;
        medium.densityMin = valueToScalar(grid->tree().root().valueMin()) * medium.densityFactor;
        medium.densityMax = medium.densityFactor; //grid->tree().root().valueMax() * medium.densityFactor;
        medium.densityMax = fmaxf(medium.densityMin, fmaxf(medium.densityMax, 0.001f));
        medium.hgMeanCosine = 0.f;

        float color[4] = {0, 0, 0, 0};

        for (int sampleIndex = 0; sampleIndex < params.samplesPerPixel; ++sampleIndex) {
            uint32_t pixelSeed = hash((sampleIndex + (numAccumulations + 1) * params.samplesPerPixel)) ^ hash(ix, iy);

            float u = ix + 0.5f;
            float v = iy + 0.5f;

            if (numAccumulations > 0 || params.samplesPerPixel > 0) {
#if 1
                float jitterX, jitterY;
                cmj(jitterX, jitterY, (sampleIndex + (numAccumulations + 1) * params.samplesPerPixel) % 64, 8, 8, pixelSeed);
                u += jitterX - 0.5f;
                v += jitterY - 0.5f;
#else
                float randVar1 = randomf(pixelSeed + 0);
                float randVar2 = randomf(pixelSeed + 1);
                u += randVar1 - 0.5f;
                v += randVar2 - 0.5f;
#endif
            }

            u /= width;
            v /= height;

            RayT wRay = camera.getRay(u, v);
            RayT iRay = wRay.worldToIndexF(*grid);

            Vec3T iRayDir = iRay.dir();
            Vec3T wRayDir = wRay.dir();
            Vec3T wRayEye = wRay.eye();

            Vec3T radiance = Vec3T(0);
            if (params.useLighting > 0) {
                radiance = traceVolume(iRay, acc, medium, pixelSeed, iLightDir);
            }

            float transmittance = getTransmittance(iRay, acc, medium, pixelSeed);

            if (transmittance > 0.01f) {
                float groundIntensity = 0.0f;
                float groundMix = 0.0f;

                if (params.useGround > 0) {
                    // intersect with ground plane and draw checker if camera is above...

                    float wGroundT = (params.groundHeight - wRayEye[1]) / wRayDir[1];

                    if (wGroundT > 0.f) {
                        Vec3T wGroundPos = wRayEye + wGroundT * wRayDir;
                        Vec3T iGroundPos = grid->worldToIndexF(wGroundPos);

                        rayTraceGround(wGroundT, params.groundFalloff, wGroundPos, wRayDir[1], groundIntensity, groundMix);

                        if (params.useShadows > 0) {
                            RayT  iShadowRay(iGroundPos, iLightDir);
                            float shadowTransmittance = getTransmittance(iShadowRay, acc, medium, pixelSeed);
                            groundIntensity *= shadowTransmittance;
                        }
                    }
                }

                float skyIntensity = 0.75f + 0.25f * wRayDir[1];

                radiance = radiance + nanovdb::Vec3f(transmittance * ((1.f - groundMix) * skyIntensity + groundMix * groundIntensity));
            }

            color[0] += radiance[0];
            color[1] += radiance[1];
            color[2] += radiance[2];
        }

        for (int k = 0; k < 3; ++k)
            color[k] = color[k] / params.samplesPerPixel;

        if (numAccumulations > 1) {
            float oldLinearPixel[3];
            if (params.useTonemapping)
                invTonemapReinhard(oldLinearPixel, outPixel, params.tonemapWhitePoint);
            else
                invTonemapPassthru(oldLinearPixel, outPixel);
            for (int k = 0; k < 3; ++k)
                color[k] = oldLinearPixel[k] + (color[k] - oldLinearPixel[k]) * (1.0f / numAccumulations);
        }

        if (params.useTonemapping)
            tonemapReinhard(outPixel, color, params.tonemapWhitePoint);
        else
            tonemapPassthru(outPixel, color);
        outPixel[3] = 1.0;
    }
};
}
} // namespace render::fogvolume
