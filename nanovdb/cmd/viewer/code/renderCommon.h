// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////

CNANOVDB_DECLARE_UNIFORMS_BEGIN
int   width;
int   height;
int   numAccumulations;
int   useBackground;
int   useGround;
int   useShadows;
int   useGroundReflections;
int   useLighting;
float useOcclusion;
float volumeDensityScale;
float volumeAlbedo;
int   useTonemapping;
float tonemapWhitePoint;
int   samplesPerPixel;
float groundHeight;
float groundFalloff;
float cameraPx, cameraPy, cameraPz;
float cameraUx, cameraUy, cameraUz;
float cameraVx, cameraVy, cameraVz;
float cameraWx, cameraWy, cameraWz;
float cameraAspect;
float cameraFovY;
CNANOVDB_DECLARE_UNIFORMS_END

////////////////////////////////////////////////////////

#define CNANOVDB_RENDERMETHOD_LEVELSET 0
#define CNANOVDB_RENDERMETHOD_FOG_VOLUME 1
#define CNANOVDB_RENDERMETHOD_GRID 2
#define CNANOVDB_RENDERMETHOD_POINTS 3

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
    return CNANOVDB_MAKE(float)(hash1(s)) / CNANOVDB_MAKE(float)(0xffffffffu);
}

CNANOVDB_INLINE vec3 makeVec3(float x)
{
    return CNANOVDB_MAKE_VEC3(x, x, x);
}

CNANOVDB_INLINE float evalSkyMaterial(const vec3 dir)
{
    return 0.75f + 0.25f * dir.y;
}

CNANOVDB_INLINE float evalGroundMaterial(float groundT, float falloffDistance, const vec3 pos, float rayDirY, CNANOVDB_REF(float) outMix)
{
    const float checkerScale = 1.0f / 1024.0f;
    float       iu = floor(pos.x * checkerScale);
    float       iv = floor(pos.z * checkerScale);
    float       outIntensity = 0.25f + fabs(fmod(iu + iv, 2.f)) * 0.5f;
    CNANOVDB_DEREF(outMix) = fmax(0.f, (1.0f - groundT / falloffDistance) * -rayDirY);
    return outIntensity;
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

CNANOVDB_INLINE Vec3T getRayDirFromPixelCoord(uint32_t ix, uint32_t iy, int width, int height, int numAccumulations, int samplesPerPixel, uint32_t pixelSeed, const Vec3T cameraU, const Vec3T cameraV, const Vec3T cameraW, float fovY, float aspect)
{
    float u = CNANOVDB_MAKE(float)(ix) + 0.5f;
    float v = CNANOVDB_MAKE(float)(iy) + 0.5f;

    float randVar1 = randomf(pixelSeed + 0);
    float randVar2 = randomf(pixelSeed + 1);

    if (numAccumulations > 0) {
        u += randVar1 - 0.5f;
        v += randVar2 - 0.5f;
    }

    u /= CNANOVDB_MAKE(float)(width);
    v /= CNANOVDB_MAKE(float)(height);

    // get camera ray...
    float halfHeight = tanf(fovY * 3.14159265358979323846f / 360.f);
    float halfWidth = aspect * halfHeight;
    vec3  W = vec3_add(vec3_add(vec3_fmul(halfWidth, cameraU), vec3_fmul(halfHeight, cameraV)), cameraW);
    vec3  U = vec3_fmul(2.f * halfWidth, cameraU);
    vec3  V = vec3_fmul(2.f * halfHeight, cameraV);
    vec3  rd = vec3_sub(vec3_add(vec3_fmul(u, U), vec3_fmul(v, V)), W);
    return vec3_normalize(rd);
}

CNANOVDB_INLINE void compositeFn(CNANOVDB_IMAGE_TYPE image, int width, ivec2 tid, vec3 color, int numAccumulations, int useTonemapping, float tonemapWhitePoint)
{
    if (numAccumulations > 1) {
        vec4 prevOutput = imageLoadPixel(image, width, tid);
        vec3 oldLinearPixel;
        vec3 prevColor = CNANOVDB_MAKE_VEC3(prevOutput.x, prevOutput.y, prevOutput.z);
        if (useTonemapping != 0) {
            invTonemapReinhard(CNANOVDB_ADDRESS(oldLinearPixel), prevColor, tonemapWhitePoint);
        } else {
            oldLinearPixel = prevColor;
        }
        color = vec3_add(oldLinearPixel, vec3_fmul((1.0f / CNANOVDB_MAKE(float)(numAccumulations)), vec3_sub(color, oldLinearPixel)));
    }

    if (useTonemapping != 0) {
        tonemapReinhard(CNANOVDB_ADDRESS(color), color, tonemapWhitePoint);
    }
    imageStorePixel(image, width, tid, CNANOVDB_MAKE_VEC4(color.x, color.y, color.z, 1.0f));
}

////////////////////////////////////////////////////////