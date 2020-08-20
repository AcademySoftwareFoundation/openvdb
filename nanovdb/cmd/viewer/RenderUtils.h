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

inline __hostdev__ void rayTraceGround(float groundT, float falloffDistance, const nanovdb::Vec3f& pos, float rayDirY, float& outIntensity, float& outMix)
{
    const float checkerScale = 1.0f / 1024.0f;

    auto iu = floorf(pos[0] * checkerScale);
    auto iv = floorf(pos[2] * checkerScale);
    outIntensity = fabsf(fmodf(iu + iv, 2.f));
    outIntensity = 0.25f + outIntensity * 0.5f;
    //float m = expf( -wGroundT / falloffDistance );// * -rayDirY;
    float m = (1.0f - groundT / falloffDistance) * -rayDirY;
    outMix = fmaxf(0.f, m);
}

// algorithm taken from: http://amietia.com/lambertnotangent.html
inline __hostdev__ nanovdb::Vec3f lambertNoTangent(nanovdb::Vec3f normal, float u, float v)
{
    float theta = 6.283185f * u;
    v = 2.0f * v - 1.0f;
    float d = sqrtf(1.0f - v * v);
    auto  spherePoint = nanovdb::Vec3f(d * cosf(theta), d * sinf(theta), v);
    return (normal + spherePoint).normalize();
}

} // namespace render