// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

//	\file renderFogVolume.c
//	\author Wil Braithwaite
//	\date July 24, 2020
//	\brief General (C99-ish) implementation of the FogVolume rendering code.
//
////////////////////////////////////////////////////////

#define CNANOVDB_VIEWER_USE_RATIO_TRACKED_TRANSMISSION 1

CNANOVDB_DECLARE_STRUCT_BEGIN(HeterogenousMedium)
    float densityFactor;
    float densityMin;
    float densityMax;
    float hgMeanCosine;
CNANOVDB_DECLARE_STRUCT_END(HeterogenousMedium)

/////////////////////////

CNANOVDB_INLINE float
deltaTracking(CNANOVDB_CONTEXT cxt, CNANOVDB_REF(nanovdb_Ray) ray, CNANOVDB_REF(nanovdb_ReadAccessor) acc, HeterogenousMedium medium, CNANOVDB_REF(uint32_t) seed)
{
    boolean hit = nanovdb_Ray_clip(ray,
                                   nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).root.mBBox_min),
                                   nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).root.mBBox_max));
    if (!hit)
        return -1.f;

    float densityMaxInv = 1.0f / medium.densityMax;
    float t = CNANOVDB_DEREF(ray).mT0;
    do {
        t += -logf(randomf(CNANOVDB_DEREF(seed)++)) * densityMaxInv;
    } while (t < CNANOVDB_DEREF(ray).mT1 && nanovdb_ReadAccessor_getValue(cxt, acc, nanovdb_Vec3fToCoord(nanovdb_Ray_eval(ray, t))) * medium.densityFactor * densityMaxInv < randomf(CNANOVDB_DEREF(seed)++));
    return t;
}

CNANOVDB_INLINE float
getTransmittance(CNANOVDB_CONTEXT cxt, CNANOVDB_REF(nanovdb_Ray) ray, CNANOVDB_REF(nanovdb_ReadAccessor) acc, HeterogenousMedium medium, CNANOVDB_REF(uint32_t) seed)
{
    boolean hit = nanovdb_Ray_clip(ray,
                                   nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).root.mBBox_min),
                                   nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).root.mBBox_max));
    if (!hit)
        return 1.0f;

#if !defined(CNANOVDB_VIEWER_USE_RATIO_TRACKED_TRANSMISSION)
    // delta tracking.
    // faster due to earlier termination, but we need multiple samples
    // to reduce variance.
    float     densityMaxInv = 1.0f / medium.densityMax;
    const int nSamples = 2;
    float     transmittance = 0.f;
    for (int n = 0; n < nSamples; n++) {
        float t = CNANOVDB_DEREF(ray).mT0;
        while (CNANOVDB_TRUE) {
            t -= logf(randomf(CNANOVDB_DEREF(seed)++)) * densityMaxInv;

            if (t >= CNANOVDB_DEREF(ray).mT1) {
                transmittance += 1.0f;
                break;
            }

            float density = nanovdb_ReadAccessor_getValue(cxt, acc, nanovdb_Vec3fToCoord(nanovdb_Ray_eval(ray, t))) * medium.densityFactor;
            density = fmin(fmax(density, medium.densityMin), medium.densityMax); // just in case these are soft extrema

            if (density * densityMaxInv >= randomf(CNANOVDB_DEREF(seed)++))
                break;
        }
    }
    return transmittance / nSamples;
#elif 0
    // ratio tracking.
    // slower due to no early termination, but better estimation.
    float densityMaxInv = 1.0f / medium.densityMax;
    float transmittance = 1.f;
    float t = CNANOVDB_DEREF(ray).mT0;
    while (CNANOVDB_TRUE) {
        t -= logf(randomf(CNANOVDB_DEREF(seed)++)) * densityMaxInv;
        if (t >= CNANOVDB_DEREF(ray).mT1)
            break;
        float density = nanovdb_ReadAccessor_getValue(cxt, acc, nanovdb_Vec3fToCoord(nanovdb_Ray_eval(ray, t))) * medium.densityFactor;
        density = fmin(fmax(density, medium.densityMin), medium.densityMax); // just in case these are soft extrema

        transmittance *= 1.0f - density * densityMaxInv;
    }
    return transmittance;
#elif 1
        // residual ratio tracking.
#if 1
    // control is minimum.
    const float controlDensity = medium.densityMin;
    const float residualDensityMax = fmax(0.001f, medium.densityMax - controlDensity);
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
    float controlTransmittance = expf(-controlDensity * (CNANOVDB_DEREF(ray).mT1 - CNANOVDB_DEREF(ray).mT0));
    float residualTransmittance = 1.f;
    float t = CNANOVDB_DEREF(ray).mT0;
    while (CNANOVDB_TRUE) {
        t -= logf(randomf(CNANOVDB_DEREF(seed)++)) * residualDensityMaxInv;
        if (t >= CNANOVDB_DEREF(ray).mT1)
            break;
        float density = nanovdb_ReadAccessor_getValue(cxt, acc, nanovdb_Vec3fToCoord(nanovdb_Ray_eval(ray, t))) * medium.densityFactor;
        density = fmin(fmax(density, medium.densityMin), medium.densityMax); // just in case these are soft extrema

        float residualDensity = density - controlDensity;
        residualTransmittance *= 1.0f - residualDensity / residualDensityMax1;
    }
    return residualTransmittance * controlTransmittance;
#endif // CNANOVDB_VIEWER_USE_RATIO_TRACKED_TRANSMISSION
}

CNANOVDB_INLINE float henyeyGreenstein(float g, float cosTheta)
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

CNANOVDB_INLINE vec3 sampleHG(float g, float e1, float e2)
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
    return CNANOVDB_MAKE_VEC3(cosf(phi) * cosTheta, sinf(phi) * sinTheta, cosTheta);
}

CNANOVDB_INLINE Vec3T estimateLight(CNANOVDB_CONTEXT cxt, Vec3T pos, Vec3T dir, CNANOVDB_REF(nanovdb_ReadAccessor) acc, HeterogenousMedium medium, CNANOVDB_REF(uint32_t) seed, Vec3T lightDir)
{
    const Vec3T lightRadiance = CNANOVDB_MAKE_VEC3(1, 1, 1);
    nanovdb_Ray shadowRay;
    shadowRay.mEye = pos;
    shadowRay.mDir = lightDir;
    shadowRay.mT0 = DeltaFloat;
    shadowRay.mT1 = MaxFloat;

    float transmittance = getTransmittance(cxt, CNANOVDB_ADDRESS(shadowRay), acc, medium, seed);
    float pdf = henyeyGreenstein(medium.hgMeanCosine, vec3_dot(dir, lightDir));
    return vec3_fmul(pdf * transmittance, lightRadiance);
}

CNANOVDB_INLINE nanovdb_Vec3f traceVolume(CNANOVDB_CONTEXT cxt, CNANOVDB_REF(nanovdb_Ray) ray, CNANOVDB_REF(nanovdb_ReadAccessor) acc, HeterogenousMedium medium, CNANOVDB_REF(uint32_t) seed, nanovdb_Vec3f lightDir)
{
    float throughput = 1.0f;

    float albedo = 0.8f;

    int max_interactions = 40;
    int num_interactions = 0;

    if (!nanovdb_Ray_clip(ray, nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).root.mBBox_min), nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).root.mBBox_max))) {
        return CNANOVDB_MAKE_VEC3(0, 0, 0);
    }

    nanovdb_Ray pathRay = CNANOVDB_DEREF(ray);

    nanovdb_Vec3f radiance = CNANOVDB_MAKE_VEC3(0, 0, 0);

    while (CNANOVDB_TRUE) {
        float s = deltaTracking(cxt, CNANOVDB_ADDRESS(pathRay), acc, medium, seed);
        if (s >= pathRay.mT1)
            break;

        if (s < 0)
            return CNANOVDB_MAKE_VEC3(0, 0, 0);

        if (num_interactions++ >= max_interactions)
            return CNANOVDB_MAKE_VEC3(0, 0, 0);

        nanovdb_Vec3f pos = nanovdb_Ray_eval(CNANOVDB_ADDRESS(pathRay), s);

        // sample key light.
        radiance = vec3_add(radiance, vec3_fmul(throughput, estimateLight(cxt, pos, pathRay.mDir, acc, medium, seed, lightDir)));

        throughput *= albedo;

        // Russian roulette absorption.
        if (throughput < 0.2f) {
            float r1 = randomf(CNANOVDB_DEREF(seed)++);
            if (r1 > throughput * 5.0f)
                return CNANOVDB_MAKE_VEC3(0, 0, 0); // full absorbtion.
            throughput = 0.2f; // unbias.
        }

        // modify ray using phase function.
        float r2 = randomf(CNANOVDB_DEREF(seed)++);
        float r3 = randomf(CNANOVDB_DEREF(seed)++);
        pathRay.mEye = pos;
        pathRay.mDir = sampleHG(medium.hgMeanCosine, r2, r3);
        pathRay.mT0 = DeltaFloat;
        pathRay.mT1 = MaxFloat;

        if (!nanovdb_Ray_clip(CNANOVDB_ADDRESS(pathRay), nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).root.mBBox_min), nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).root.mBBox_max)))
            return CNANOVDB_MAKE_VEC3(0, 0, 0);
    }
    /*
	const float f = (0.5f + 0.5f * ray.dir()[1]) * throughput;
	radiance = radiance + Vec3f(f);*/

    return radiance;
}

/////////////////////////

#if defined(CNANOVDB_COMPILER_GLSL)

layout(rgba32f, binding = 0) uniform image2D outImage;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    int cxt;

    ivec2 threadId = getThreadId();

#elif defined(__OPENCL_VERSION__)

CNANOVDB_KERNEL void renderFogVolume(
    CNANOVDB_GLOBAL vec4* outImage,
    CNANOVDB_GLOBAL const nanovdb_Node0_float* nodeLevel0,
    CNANOVDB_GLOBAL const nanovdb_Node1_float* nodeLevel1,
    CNANOVDB_GLOBAL const nanovdb_Node2_float* nodeLevel2,
    CNANOVDB_GLOBAL const nanovdb_RootData_float* rootData,
    CNANOVDB_GLOBAL const nanovdb_RootData_Tile_float* rootDataTiles,
    CNANOVDB_GLOBAL const nanovdb_GridData* gridData,
    ArgUniforms kArgs)
{
    TreeContext _cxt;
    _cxt.mNodeLevel0.nodes = nodeLevel0;
    _cxt.mNodeLevel1.nodes = nodeLevel1;
    _cxt.mNodeLevel2.nodes = nodeLevel2;
    _cxt.mRootData.root = *rootData;
    _cxt.mRootData.tiles = rootDataTiles;
    _cxt.mGridData.grid = *gridData;
    const TreeContext* cxt = &_cxt;

    ivec2 threadId = getThreadId();
#else

CNANOVDB_KERNEL void renderFogVolume(
    ivec2 threadId,
    CNANOVDB_GLOBAL vec4* outImage,
    CNANOVDB_GLOBAL const nanovdb_Node0_float* nodeLevel0,
    CNANOVDB_GLOBAL const nanovdb_Node1_float* nodeLevel1,
    CNANOVDB_GLOBAL const nanovdb_Node2_float* nodeLevel2,
    CNANOVDB_GLOBAL const nanovdb_RootData_float* rootData,
    CNANOVDB_GLOBAL const nanovdb_RootData_Tile_float* rootDataTiles,
    CNANOVDB_GLOBAL const nanovdb_GridData* gridData,
    ArgUniforms kArgs)
{
    TreeContext _cxt;
    _cxt.mNodeLevel0.nodes = nodeLevel0;
    _cxt.mNodeLevel1.nodes = nodeLevel1;
    _cxt.mNodeLevel2.nodes = nodeLevel2;
    _cxt.mRootData.root = *rootData;
    _cxt.mRootData.tiles = rootDataTiles;
    _cxt.mGridData.grid = *gridData;
    const TreeContext* cxt = &_cxt;
#endif

    uint32_t ix = threadId.x;
    uint32_t iy = threadId.y;

    if (ix >= (uint32_t)kArgs.width || iy >= (uint32_t)kArgs.height)
        return;

    vec3 cameraP = CNANOVDB_MAKE_VEC3(kArgs.cameraPx, kArgs.cameraPy, kArgs.cameraPz);
    vec3 cameraU = CNANOVDB_MAKE_VEC3(kArgs.cameraUx, kArgs.cameraUy, kArgs.cameraUz);
    vec3 cameraV = CNANOVDB_MAKE_VEC3(kArgs.cameraVx, kArgs.cameraVy, kArgs.cameraVz);
    vec3 cameraW = CNANOVDB_MAKE_VEC3(kArgs.cameraWx, kArgs.cameraWy, kArgs.cameraWz);

    const vec3 wLightDir = CNANOVDB_MAKE_VEC3(0.0f, 1.0f, 0.0f);
    const vec3 iLightDir = vec3_normalize(nanovdb_Grid_worldToIndexDirF(CNANOVDB_GRIDDATA(cxt).grid, wLightDir));

#if 0
    {
        vec4 color = CNANOVDB_MAKE_VEC4(fabs(cameraV.x), fabs(cameraV.y), fabs(cameraV.z), 1);
        imageStorePixel(outImage, kArgs.width, threadId, color);
        return;
    }
#endif

    nanovdb_ReadAccessor acc = nanovdb_ReadAccessor_create();
    vec3                 color = CNANOVDB_MAKE_VEC3(0, 0, 0);

    HeterogenousMedium medium;
    medium.densityFactor = kArgs.volumeDensity;
    medium.densityMin = CNANOVDB_ROOTDATA(cxt).root.mValueMin * medium.densityFactor;
    medium.densityMax = medium.densityFactor; //grid->tree().root().valueMax() * medium.densityFactor;
    medium.densityMax = fmax(medium.densityMin, fmax(medium.densityMax, 0.001f));
    medium.hgMeanCosine = 0.f;

    
    for (int sampleIndex = 0; sampleIndex < kArgs.samplesPerPixel; ++sampleIndex) {
        uint32_t pixelSeed = hash1(sampleIndex + kArgs.numAccumulations * kArgs.samplesPerPixel ^ hash2(ix, iy));

        float u = (float)(ix) + 0.5f;
        float v = (float)(iy) + 0.5f;

        float randVar1 = randomf(pixelSeed + 0);
        float randVar2 = randomf(pixelSeed + 1);

        if (kArgs.numAccumulations > 0) {
            u += randVar1 - 0.5f;
            v += randVar2 - 0.5f;
        }

        u /= (float)(kArgs.width);
        v /= (float)(kArgs.height);

        // get camera ray...
        vec3        wRayDir = vec3_sub(vec3_add(vec3_fmul(u, cameraU), vec3_fmul(v, cameraV)), cameraW);
        vec3        wRayEye = cameraP;
        nanovdb_Ray wRay;
        wRay.mEye = wRayEye;
        wRay.mDir = wRayDir;
        wRay.mT0 = 0;
        wRay.mT1 = MaxFloat;

        nanovdb_Ray iRay = nanovdb_Ray_worldToIndexF(wRay, CNANOVDB_GRIDDATA(cxt).grid);
        vec3        iRayDir = iRay.mDir;

        Vec3T radiance = CNANOVDB_MAKE_VEC3(0, 0, 0);

        if (kArgs.useLighting > 0) {
            radiance = traceVolume(cxt, CNANOVDB_ADDRESS(iRay), CNANOVDB_ADDRESS(acc), medium, CNANOVDB_ADDRESS(pixelSeed), iLightDir);
        }

        float transmittance = getTransmittance(cxt, CNANOVDB_ADDRESS(iRay), CNANOVDB_ADDRESS(acc), medium, CNANOVDB_ADDRESS(pixelSeed));

        if (transmittance > 0.01f) {
            float groundIntensity = 0.0f;
            float groundMix = 0.0f;

            if (kArgs.useGround > 0) {
                float wGroundT = (kArgs.groundHeight - wRayEye.y) / wRayDir.y;
                if (wGroundT > 0.f) {
                    vec3 wGroundPos = vec3_add(wRayEye, vec3_fmul(wGroundT, wRayDir));
                    vec3 iGroundPos = nanovdb_Grid_worldToIndexF(CNANOVDB_GRIDDATA(cxt).grid, wGroundPos);

                    rayTraceGround(wGroundT, kArgs.groundFalloff, wGroundPos, wRayDir.y, CNANOVDB_ADDRESS(groundIntensity), CNANOVDB_ADDRESS(groundMix));

                    if (kArgs.useShadows > 0) {
                        nanovdb_Ray iShadowRay;
                        iShadowRay.mEye = iGroundPos;
                        iShadowRay.mDir = iLightDir;
                        iShadowRay.mT0 = DeltaFloat;
                        iShadowRay.mT1 = MaxFloat;

                        float shadowTransmittance = getTransmittance(cxt, CNANOVDB_ADDRESS(iShadowRay), CNANOVDB_ADDRESS(acc), medium, CNANOVDB_ADDRESS(pixelSeed));
                        groundIntensity *= shadowTransmittance;
                    }
                }
            }

            float skyIntensity = 0.75f + 0.25f * wRayDir.y;

            float radianceIntensity = transmittance * ((1.f - groundMix) * skyIntensity + groundMix * groundIntensity);
            radiance = vec3_add(radiance, CNANOVDB_MAKE_VEC3(radianceIntensity, radianceIntensity, radianceIntensity));
        }

        color = vec3_add(color, radiance);
    }

    color.x /= kArgs.samplesPerPixel;
    color.y /= kArgs.samplesPerPixel;
    color.z /= kArgs.samplesPerPixel;

    if (kArgs.numAccumulations > 1) {
        vec4 prevOutput = imageLoadPixel(outImage, kArgs.width, threadId);
        vec3 prevColor = CNANOVDB_MAKE_VEC3(prevOutput.x, prevOutput.y, prevOutput.z);
        vec3 oldLinearPixel;
        invTonemapReinhard(CNANOVDB_ADDRESS(oldLinearPixel), prevColor, kArgs.tonemapWhitePoint);
        color = vec3_add(oldLinearPixel, vec3_fmul((1.0f / (float)(kArgs.numAccumulations)), vec3_sub(color, oldLinearPixel)));
    }

    tonemapReinhard(CNANOVDB_ADDRESS(color), color, kArgs.tonemapWhitePoint);
    imageStorePixel(outImage, kArgs.width, threadId, CNANOVDB_MAKE_VEC4(color.x, color.y, color.z, 1.0f));
}
////////////////////////////////////////////////////////