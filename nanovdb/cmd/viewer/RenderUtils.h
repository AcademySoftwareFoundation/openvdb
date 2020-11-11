// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderFunctions.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief General C++ implementation rendering utilities code.
*/

#pragma once

#include <nanovdb/util/Ray.h>

namespace render {

template<typename Vec3T1, typename Vec3T2>
inline __hostdev__ void tonemapPassthru(Vec3T1& out, Vec3T2& in)
{
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
}

template<typename Vec3T1, typename Vec3T2>
inline __hostdev__ void invTonemapPassthru(Vec3T1& out, Vec3T2& in)
{
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
}

inline __hostdev__ float reinhardFn(float x)
{
    return x / (1.0f + x);
}

inline __hostdev__ float invReinhardFn(float x)
{
    return x / (1.0f - x);
}

template<typename Vec3T1, typename Vec3T2>
inline __hostdev__ void tonemapReinhard(Vec3T1& out, Vec3T2& in, const float W = 10)
{
    const float invW = 1.0f / reinhardFn(W);
    out[0] = reinhardFn(in[0]) * invW;
    out[1] = reinhardFn(in[1]) * invW;
    out[2] = reinhardFn(in[2]) * invW;
}

template<typename Vec3T1, typename Vec3T2>
inline __hostdev__ void invTonemapReinhard(Vec3T1& out, Vec3T2& in, const float W = 10)
{
    float w = reinhardFn(W);
    out[0] = invReinhardFn(in[0] * w);
    out[1] = invReinhardFn(in[1] * w);
    out[2] = invReinhardFn(in[2] * w);
}

template<typename ValueT, typename Vec3T>
inline __hostdev__ ValueT luminance(Vec3T v)
{
    return ValueT(v[0] * ValueT(0.2126) + v[1] * ValueT(0.7152) + v[2] * ValueT(0.0722));
}

template<typename Vec3T>
inline __hostdev__ void tonemapACES(Vec3T& out, const Vec3T& in)
{
    const float x0 = 2.51f;
    const float x1 = 0.03f;
    const float x2 = 2.43f;
    const float x3 = 0.59f;
    const float x4 = 0.14f;
    const float r = (in[0] * (x0 * in[0] + x1)) / (in[0] * (x2 * in[0] + x3) + x4);
    const float g = (in[1] * (x0 * in[1] + x1)) / (in[1] * (x2 * in[1] + x3) + x4);
    const float b = (in[2] * (x0 * in[2] + x1)) / (in[2] * (x2 * in[2] + x3) + x4);
    out[0] = nanovdb::Max(nanovdb::Min(r, 1.0f), 0.0f);
    out[1] = nanovdb::Max(nanovdb::Min(g, 1.0f), 0.0f);
    out[2] = nanovdb::Max(nanovdb::Min(b, 1.0f), 0.0f);
}

// LCG values from Numerical Recipes
inline __hostdev__ float randomLCG(uint32_t& seed)
{
    seed = 1664525 * seed + 1013904223;
    return seed / float(0xffffffffu);
}

// Xorshift algorithm from George Marsaglia
inline __hostdev__ float randomXorShift(uint32_t& seed)
{
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return seed / float(0xffffffffu);
}

// http://www.burtleburtle.net/bob/hash/doobs.html
inline __hostdev__ uint32_t hash(uint32_t x)
{
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

inline __hostdev__ uint32_t hash(uint32_t x, uint32_t y)
{
    return hash(x ^ hash(y));
}

inline __hostdev__ float randomf(uint32_t s)
{
    return hash(s) / float(0xffffffffu);
}

inline __hostdev__ unsigned permute(unsigned seed, unsigned range)
{
    seed ^= unsigned(seed >> 19);
    seed *= unsigned(1619700021);
    seed ^= unsigned(seed >> 16);
    seed *= unsigned(175973783);
    seed ^= unsigned(seed >> 15);

    unsigned f = seed & unsigned(0x0000ffff);
    return (range * f + (seed % range)) >> 16;
}

inline __hostdev__ void cmj(float& outX, float& outY, int s, int w, int h, int seed)
{
    auto sx = float(permute(seed * 2 + 0, 256)) / 256.f;
    auto sy = float(permute(seed * 2 + 1, 256)) / 256.f;
    outX = (s % w + sy) / w;
    outY = (s / w + sx) / h;
}

inline __hostdev__ nanovdb::Ray<float> getRayFromPixelCoord(int ix, int iy, int width, int height, const SceneRenderParameters& sceneParams)
{
    float u = ix + 0.5f;
    float v = iy + 0.5f;
    return sceneParams.camera.getRay(u / width, v / height);
}

inline __hostdev__ nanovdb::Ray<float> getRayFromPixelCoord(int ix, int iy, int width, int height, int numAccumulations, int samplesPerPixel, uint32_t& pixelSeed, const SceneRenderParameters& sceneParams)
{
    float u = ix + 0.5f;
    float v = iy + 0.5f;

    if (numAccumulations > 0 || samplesPerPixel > 0) {
#if 1
        float jitterX, jitterY;
        cmj(jitterX, jitterY, numAccumulations % 64, 8, 8, pixelSeed);
        u += jitterX - 0.5f;
        v += jitterY - 0.5f;
#else
        float randVar1 = randomf(pixelSeed + 0);
        float randVar2 = randomf(pixelSeed + 1);
        u += randVar1 - 0.5f;
        v += randVar2 - 0.5f;
#endif
    }

    return sceneParams.camera.getRay(u / width, v / height);
}

inline __hostdev__ float smoothstep(float t, float edge0, float edge1)
{
    t = nanovdb::Clamp((t - edge0) / (edge1 - edge0), 0.f, 1.f);
    return t * t * (3.0f - 2.0f * t);
}

inline __hostdev__ float evalGroundMaterial(float wGroundT, float wFalloffDistance, const nanovdb::Vec3f& pos, float rayDirY, float& outMix)
{
    const float s = nanovdb::Min(wGroundT / wFalloffDistance, 1.f);

    outMix = nanovdb::Max(0.f, (1.0f - s) * -rayDirY);

    static constexpr float checkerScale = 1.0f / float(1 << (3 + 4));
    auto                   iu = floorf(pos[0] * checkerScale);
    auto                   iv = floorf(pos[2] * checkerScale);
    float                  floorIntensity = 0.25f + fabsf(fmodf(iu + iv, 2.f)) * 0.5f;

    float       t = nanovdb::Max(fmodf(nanovdb::Abs(pos[0]), 1.0f), fmodf(nanovdb::Abs(pos[2]), 1.0f));
    const float lineWidth = s;
    float       grid = smoothstep(t, 0.97f - lineWidth, 1.0f - lineWidth);
    // fade the grid out before the checkboard floor. (This avoids quite a lot of aliasing)
    grid *= nanovdb::Max(0.f, (1.0f - nanovdb::Min((wGroundT / 100.f), 1.0f)) * -rayDirY);

    return floorIntensity + grid;
}

inline __hostdev__ float evalSkyMaterial(const nanovdb::Vec3f& dir)
{
    return 0.75f + 0.25f * dir[1];
}

__hostdev__ inline float traceEnvironment(const nanovdb::Ray<float>& wRay, const SceneRenderParameters& sceneParams)
{
    if (!sceneParams.useBackground)
        return 0.0f;

    float skyIntensity = evalSkyMaterial(wRay.dir());

    if (!sceneParams.useGround)
        return skyIntensity;

    float groundIntensity = 0.0f;
    float groundMix = 0.0f;
    if (sceneParams.useGround) {
        float wGroundT = (sceneParams.groundHeight - wRay.eye()[1]) / wRay.dir()[1];
        if (wRay.dir()[1] != 0 && wGroundT > 0.f) {
            nanovdb::Vec3f wGroundPos = wRay.eye() + wGroundT * wRay.dir();
            groundIntensity = evalGroundMaterial(wGroundT, sceneParams.groundFalloff, wGroundPos, wRay.dir()[1], groundMix);
        }
    }
    float bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
    return bgIntensity;
}

// algorithm adapted from: https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Utils/Math/MathHelpers.slang
// Generate a vector that is orthogonal to the input vector.
inline __hostdev__ nanovdb::Vec3f perpStark(const nanovdb::Vec3f& u)
{
    auto a = nanovdb::Vec3f(nanovdb::Abs(u[0]), nanovdb::Abs(u[1]), nanovdb::Abs(u[2]));
    auto uyx = (a[0] - a[1]) < 0 ? 1 : 0;
    auto uzx = (a[0] - a[2]) < 0 ? 1 : 0;
    auto uzy = (a[1] - a[2]) < 0 ? 1 : 0;
    auto xm = uyx & uzx;
    auto ym = (1 ^ xm) & uzy;
    auto zm = 1 ^ (xm | ym); // 1 ^ (xm & ym)
    auto v = u.cross(nanovdb::Vec3f(float(xm), float(ym), float(zm)));
    return v;
}

// algorithm taken from: http://amietia.com/lambertnotangent.html
// Return a vector in the cosine distribution.
inline __hostdev__ nanovdb::Vec3f lambertNoTangent(nanovdb::Vec3f normal, float u, float v)
{
    float theta = 6.283185f * u;
    v = 2.0f * v - 1.0f;
    float d = sqrtf(1.0f - v * v);
    auto  spherePoint = nanovdb::Vec3f(d * cosf(theta), d * sinf(theta), v);
    return (normal + spherePoint).normalize();
}

inline __hostdev__ void compositeFn(float* outPixel, float* color, int numAccumulations, const SceneRenderParameters& sceneParams)
{
    if (numAccumulations > 1) {
        float oldLinearPixel[3];
        if (sceneParams.useTonemapping)
            invTonemapReinhard(oldLinearPixel, outPixel, sceneParams.tonemapWhitePoint);
        else
            invTonemapPassthru(oldLinearPixel, outPixel);
        for (int k = 0; k < 3; ++k)
            color[k] = oldLinearPixel[k] + (color[k] - oldLinearPixel[k]) * (1.0f / numAccumulations);
    }

    if (sceneParams.useTonemapping)
        tonemapReinhard(outPixel, color, sceneParams.tonemapWhitePoint);
    else
        tonemapPassthru(outPixel, color);
    outPixel[3] = 1.0;
}

struct RenderEnvRgba32fFn
{
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int numAccumulations, const SceneRenderParameters sceneParams, const MaterialParameters /*materialParams*/) const
    {
        auto wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);
        auto envRadiance = traceEnvironment(wRay, sceneParams);

        float color[4];
        color[0] = envRadiance;
        color[1] = envRadiance;
        color[2] = envRadiance;

        auto outPixel = &imgBuffer[4 * (ix + width * iy)];
        compositeFn(outPixel, color, numAccumulations, sceneParams);
    }
};

struct CameraDiagnosticRenderer
{
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int /*numAccumulations*/, const SceneRenderParameters sceneParams, const MaterialParameters /*materialParams*/) const
    {
        auto outPixel = &imgBuffer[4 * (ix + width * iy)];
        auto wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);

        float color[4];
        color[0] = wRay.dir()[0];
        color[1] = wRay.dir()[1];
        color[2] = wRay.dir()[2];
        color[3] = 1;

        outPixel[0] = color[0];
        outPixel[1] = color[1];
        outPixel[2] = color[2];
        outPixel[3] = color[3];
    }
};

} // namespace render