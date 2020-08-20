// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////

CNANOVDB_DECLARE_UNIFORMS_BEGIN()
int   width;
int   height;
int   numAccumulations;
float useGround;
float useShadows;
float useGroundReflections;
float useLighting;
float useOcclusion;
float volumeDensity;
float tonemapWhitePoint;
int   samplesPerPixel;
float groundHeight;
float groundFalloff;
float cameraPx, cameraPy, cameraPz;
float cameraUx, cameraUy, cameraUz;
float cameraVx, cameraVy, cameraVz;
float cameraWx, cameraWy, cameraWz;
CNANOVDB_DECLARE_UNIFORMS_END()

////////////////////////////////////////////////////////

#define CNANOVDB_RENDERMETHOD_AUTO 0
#define CNANOVDB_RENDERMETHOD_LEVELSET 1
#define CNANOVDB_RENDERMETHOD_FOG_VOLUME 2
#define CNANOVDB_RENDERMETHOD_GRID 3
#define CNANOVDB_RENDERMETHOD_POINTS 4
#define CNANOVDB_RENDERMETHOD_COUNT 5

////////////////////////////////////////////////////////

CNANOVDB_INLINE float reinhardFn(float x)
{
    return x / (1.0f + x);
}

CNANOVDB_INLINE float invReinhardFn(float x)
{
    return x / (1.0f - x);
}

CNANOVDB_INLINE void tonemapReinhard(CNANOVDB_REF(vec3) outColor, vec3 inColor, float W)
{
    const float invW = 1.0f / reinhardFn(W);
    CNANOVDB_DEREF(outColor) = CNANOVDB_MAKE_VEC3(reinhardFn(inColor.x) * invW,
                                                  reinhardFn(inColor.y) * invW,
                                                  reinhardFn(inColor.z) * invW);
}

CNANOVDB_INLINE void invTonemapReinhard(CNANOVDB_REF(vec3) outColor, vec3 inColor, float W)
{
    const float w = reinhardFn(W);
    CNANOVDB_DEREF(outColor) = CNANOVDB_MAKE_VEC3(invReinhardFn(inColor.x * w),
                                                  invReinhardFn(inColor.y * w),
                                                  invReinhardFn(inColor.z * w));
}

// http://www.burtleburtle.net/bob/hash/doobs.html
CNANOVDB_INLINE uint32_t hash1(uint32_t x)
{
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

CNANOVDB_INLINE uint32_t hash2(uint32_t x, uint32_t y)
{
    return hash1(x ^ hash1(y));
}

CNANOVDB_INLINE float randomf(uint32_t s)
{
    return (float)(hash1(s)) / (float)(0xffffffffu);
}

CNANOVDB_INLINE vec3 makeVec3(float x)
{
    return CNANOVDB_MAKE_VEC3(x, x, x);
}

CNANOVDB_INLINE void rayTraceGround(float groundT, float falloffDistance, const vec3 pos, float rayDirY, CNANOVDB_REF(float) outIntensity, CNANOVDB_REF(float) outMix)
{
    const float checkerScale = 1.0f / 1024.0f;
    float       iu = floor(pos.x * checkerScale);
    float       iv = floor(pos.z * checkerScale);
    CNANOVDB_DEREF(outIntensity) = 0.25f + fabs(fmod(iu + iv, 2.f)) * 0.5f;
    CNANOVDB_DEREF(outMix) = fmax(0.f, (1.0f - groundT / falloffDistance) * -rayDirY);
}

// algorithm taken from: http://amietia.com/lambertnotangent.html
CNANOVDB_INLINE Vec3T lambertNoTangent(Vec3T normal, float u, float v)
{
    float theta = 6.283185f * u;
    v = 2.0f * v - 1.0f;
    float d = sqrtf(1.0f - v * v);
    Vec3T spherePoint = CNANOVDB_MAKE_VEC3(d * cosf(theta), d * sinf(theta), v);
    return vec3_normalize(vec3_add(normal, spherePoint));
}

////////////////////////////////////////////////////////