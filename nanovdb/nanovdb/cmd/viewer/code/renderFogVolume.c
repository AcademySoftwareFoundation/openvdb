// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

//	\file renderFogVolume.c
//	\brief General (C99-ish) implementation of the FogVolume rendering code.
//
////////////////////////////////////////////////////////

#if defined(CNANOVDB_COMPILER_GLSL)
#line 12
#endif

//#define CNANOVDB_VIEWER_USE_RATIO_TRACKED_TRANSMISSION 1

CNANOVDB_DECLARE_STRUCT_BEGIN(HeterogenousMedium)
    float densityScale;
    float densityMin;
    float densityMax;
    float hgMeanCosine;
    float albedo;
CNANOVDB_DECLARE_STRUCT_END(HeterogenousMedium)

/////////////////////////

CNANOVDB_INLINE float
deltaTracking(CNANOVDB_CONTEXT cxt, CNANOVDB_REF(nanovdb_Ray) ray, CNANOVDB_REF(nanovdb_ReadAccessor) acc, HeterogenousMedium medium, CNANOVDB_REF(uint32_t) seed)
{
    float densityMaxInv = 1.0f / medium.densityMax;
    float t = CNANOVDB_DEREF(ray).mT0;
    do {
        t += -logf(randomf(CNANOVDB_DEREF(seed)++)) * densityMaxInv;
    } while (t < CNANOVDB_DEREF(ray).mT1 && nanovdb_ReadAccessor_getValue(cxt, acc, nanovdb_Vec3fToCoord(nanovdb_Ray_eval(ray, t))) * medium.densityScale * densityMaxInv < randomf(CNANOVDB_DEREF(seed)++));
    return t;
}

CNANOVDB_INLINE float
getTransmittance(CNANOVDB_CONTEXT cxt, CNANOVDB_REF(nanovdb_Ray) ray, CNANOVDB_REF(nanovdb_ReadAccessor) acc, HeterogenousMedium medium, CNANOVDB_REF(uint32_t) seed)
{
    boolean hit = nanovdb_Ray_clip(ray,
                                   nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).mBBox_min),
                                   nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).mBBox_max));
    if (!hit)
        return 1.0f;

#if defined(CNANOVDB_VIEWER_USE_RATIO_TRACKED_TRANSMISSION)
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
    float       controlTransmittance = expf(-controlDensity * (CNANOVDB_DEREF(ray).mT1 - CNANOVDB_DEREF(ray).mT0));
    float       residualTransmittance = 1.f;
    float       t = CNANOVDB_DEREF(ray).mT0;
    while (CNANOVDB_TRUE) {
        t -= logf(randomf(CNANOVDB_DEREF(seed)++)) * residualDensityMaxInv;
        if (t >= CNANOVDB_DEREF(ray).mT1)
            break;
        float density = nanovdb_ReadAccessor_getValue(cxt, acc, nanovdb_Vec3fToCoord(nanovdb_Ray_eval(ray, t))) * medium.densityScale;
        //density = fmin(fmax(density, medium.densityMin), medium.densityMax); // just in case these are soft extrema

        float residualDensity = density - controlDensity;
        residualTransmittance *= 1.0f - residualDensity / residualDensityMax1;
    }
    return residualTransmittance * controlTransmittance;
#elif 1
    // ratio tracking.
    // slower due to no early termination, but better estimation.
    float densityMaxInv = 1.0f / medium.densityMax;
    float transmittance = 1.f;
    float t = CNANOVDB_DEREF(ray).mT0;
    while (CNANOVDB_TRUE) {
        t -= logf(randomf(CNANOVDB_DEREF(seed)++)) * densityMaxInv;
        if (t >= CNANOVDB_DEREF(ray).mT1)
            break;
        float density = nanovdb_ReadAccessor_getValue(cxt, acc, nanovdb_Vec3fToCoord(nanovdb_Ray_eval(ray, t))) * medium.densityScale;

        transmittance *= 1.0f - density * densityMaxInv;
        if (transmittance < 0.01f)
            return 0.f;
    }
    return transmittance;
#elif 1

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

            float density = nanovdb_ReadAccessor_getValue(cxt, acc, nanovdb_Vec3fToCoord(nanovdb_Ray_eval(ray, t))) * medium.densityScale;

            if (density * densityMaxInv >= randomf(CNANOVDB_DEREF(seed)++))
                break;
        }
    }
    return transmittance / nSamples;
#endif // CNANOVDB_VIEWER_USE_RATIO_TRACKED_TRANSMISSION
}

CNANOVDB_INLINE float henyeyGreenstein(float g, float cosTheta)
{
    if (g == 0) {
        return 1.0f / (3.14159265359f * 4.f);
    } else {
        float denom = fmax(0.001f, 1.f + g * g - 2.f * g * cosTheta);
        return (1.0f / (3.14159265359f * 4.f)) * (1.f - g * g) / (denom * sqrtf(denom));
    }
}

CNANOVDB_INLINE vec3 sampleHG(float g, float e1, float e2)
{
    // phase function.
    if (g == 0) {
        // isotropic
        const float phi = CNANOVDB_MAKE(float)(2.0f * 3.14159265359f) * e1;
        const float cosTheta = 1.0f - 2.0f * e2;
        const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
        return CNANOVDB_MAKE_VEC3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
    } else {
        const float phi = CNANOVDB_MAKE(float)(2.0f * 3.14159265359f) * e2;
        const float s = 2.0f * e1 - 1.0f;
        const float denom = fmax(0.001f, (1.0f + g * s));
        const float f = (1.0f - g * g) / denom;
        const float cosTheta = 0.5f * (1.0f / g) * (1.0f + g * g - f * f);
        const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
        return CNANOVDB_MAKE_VEC3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
    }
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
    float pdf = 1.0f; //henyeyGreenstein(medium.hgMeanCosine, vec3_dot(dir, lightDir));
    return vec3_fmul(pdf * transmittance, lightRadiance);
}

CNANOVDB_INLINE nanovdb_Vec3f traceFogVolume(CNANOVDB_CONTEXT cxt, CNANOVDB_REF(float) outThroughput, int useLighting, CNANOVDB_REF(nanovdb_Ray) ray, CNANOVDB_REF(nanovdb_ReadAccessor) acc, HeterogenousMedium medium, CNANOVDB_REF(uint32_t) seed, nanovdb_Vec3f lightDir)
{
    float throughput = 1.0f;

    int kMaxPathDepth = 4;
    int numInteractions = 0;

    if (!nanovdb_Ray_clip(ray, nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).mBBox_min), nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).mBBox_max))) {
        return CNANOVDB_MAKE_VEC3(0, 0, 0);
    }

    nanovdb_Ray pathRay = CNANOVDB_DEREF(ray);

    nanovdb_Vec3f radiance = CNANOVDB_MAKE_VEC3(0, 0, 0);

    while (numInteractions++ < kMaxPathDepth) {
        float s = deltaTracking(cxt, CNANOVDB_ADDRESS(pathRay), acc, medium, seed);
        if (s >= pathRay.mT1)
            break;

        if (s < 0) {
            radiance = CNANOVDB_MAKE_VEC3(0, 0, 0);
            break;
        }

        nanovdb_Vec3f pos = nanovdb_Ray_eval(CNANOVDB_ADDRESS(pathRay), s);

        // sample key light.
        if (useLighting > 0) {
            radiance = vec3_add(radiance, vec3_fmul(throughput, estimateLight(cxt, pos, pathRay.mDir, acc, medium, seed, lightDir)));
        }
        
        throughput *= medium.albedo;

        // Russian roulette absorption.
        if (throughput < 0.2f) {
            float r1 = randomf(CNANOVDB_DEREF(seed)++);
            if (r1 > throughput * 5.0f) {
                radiance = CNANOVDB_MAKE_VEC3(0, 0, 0);
                break;
            }
            throughput = 0.2f; // unbias.
        }

        // modify ray using phase function.
        float r2 = randomf(CNANOVDB_DEREF(seed)++);
        float r3 = randomf(CNANOVDB_DEREF(seed)++);
        pathRay.mEye = pos;
        pathRay.mDir = sampleHG(medium.hgMeanCosine, r2, r3);
        pathRay.mT0 = DeltaFloat;
        pathRay.mT1 = MaxFloat;

        if (!nanovdb_Ray_clip(CNANOVDB_ADDRESS(pathRay), nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).mBBox_min), nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).mBBox_max))) {
            radiance = CNANOVDB_MAKE_VEC3(0, 0, 0);
            break;
        }
    }

    CNANOVDB_DEREF(outThroughput) = throughput;
    return radiance;
}

/////////////////////////

#if defined(CNANOVDB_COMPILER_GLSL)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    int   cxt;
    int   outImage;
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

    if (ix >= CNANOVDB_MAKE(uint32_t)(kArgs.width) || iy >= CNANOVDB_MAKE(uint32_t)(kArgs.height))
        return;

    vec3 cameraP = CNANOVDB_MAKE_VEC3(kArgs.cameraPx, kArgs.cameraPy, kArgs.cameraPz);
    vec3 cameraU = CNANOVDB_MAKE_VEC3(kArgs.cameraUx, kArgs.cameraUy, kArgs.cameraUz);
    vec3 cameraV = CNANOVDB_MAKE_VEC3(kArgs.cameraVx, kArgs.cameraVy, kArgs.cameraVz);
    vec3 cameraW = CNANOVDB_MAKE_VEC3(kArgs.cameraWx, kArgs.cameraWy, kArgs.cameraWz);

    const vec3 wLightDir = CNANOVDB_MAKE_VEC3(0.0f, 1.0f, 0.0f);
    const vec3 iLightDir = vec3_normalize(nanovdb_Grid_worldToIndexDirF(CNANOVDB_GRIDDATA(cxt), wLightDir));

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
    medium.densityScale = kArgs.volumeDensityScale;
    medium.densityMin = CNANOVDB_ROOTDATA(cxt).mValueMin * medium.densityScale;
    medium.densityMax = medium.densityScale; //grid->tree().root().valueMax() * medium.densityScale;
    medium.densityMax = fmax(medium.densityMin, fmax(medium.densityMax, 0.001f));
    medium.hgMeanCosine = 0.f;
    medium.albedo = kArgs.volumeAlbedo;

    for (int sampleIndex = 0; sampleIndex < kArgs.samplesPerPixel; ++sampleIndex) {
        uint32_t pixelSeed = hash1(sampleIndex + kArgs.numAccumulations * kArgs.samplesPerPixel ^ hash2(ix, iy));

        vec3 wRayDir = getRayDirFromPixelCoord(ix, iy, kArgs.width, kArgs.height, kArgs.numAccumulations, pixelSeed, cameraU, cameraV, cameraW, kArgs.cameraFovY, kArgs.cameraAspect);

        vec3        wRayEye = cameraP;
        nanovdb_Ray wRay;
        wRay.mEye = wRayEye;
        wRay.mDir = wRayDir;
        wRay.mT0 = 0;
        wRay.mT1 = MaxFloat;

        nanovdb_Ray iRay = nanovdb_Ray_worldToIndexF(wRay, CNANOVDB_GRIDDATA(cxt));
        vec3        iRayDir = iRay.mDir;

        Vec3T radiance = CNANOVDB_MAKE_VEC3(0, 0, 0);
        float pathThroughput = 1.0f;

        radiance = traceFogVolume(cxt, CNANOVDB_ADDRESS(pathThroughput), kArgs.useLighting, CNANOVDB_ADDRESS(iRay), CNANOVDB_ADDRESS(acc), medium, CNANOVDB_ADDRESS(pixelSeed), iLightDir);

        //float pathThroughput = getTransmittance(cxt, CNANOVDB_ADDRESS(iRay), CNANOVDB_ADDRESS(acc), medium, CNANOVDB_ADDRESS(pixelSeed));

        if (kArgs.useBackground > 0 && pathThroughput > 0.0f) {
            float bgIntensity = 0.0f;
            float groundIntensity = 0.0f;
            float groundMix = 0.0f;

            if (kArgs.useGround > 0) {
                float wGroundT = (kArgs.groundHeight - wRayEye.y) / wRayDir.y;
                if (wRayDir.y != 0 && wGroundT > 0.f) {
                    vec3 wGroundPos = vec3_add(wRayEye, vec3_fmul(wGroundT, wRayDir));
                    vec3 iGroundPos = nanovdb_Grid_worldToIndexF(CNANOVDB_GRIDDATA(cxt), wGroundPos);

                    groundIntensity = evalGroundMaterial(wGroundT, kArgs.groundFalloff, wGroundPos, wRayDir.y, CNANOVDB_ADDRESS(groundMix));

                    if (kArgs.useLighting > 0 && kArgs.useShadows > 0) {
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

            float skyIntensity = evalSkyMaterial(wRayDir);
            bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
            bgIntensity *= pathThroughput;
            
            radiance = vec3_add(radiance, CNANOVDB_MAKE_VEC3(bgIntensity, bgIntensity, bgIntensity));
        }

        color = vec3_add(color, radiance);
    }

    color.x /= kArgs.samplesPerPixel;
    color.y /= kArgs.samplesPerPixel;
    color.z /= kArgs.samplesPerPixel;

    compositeFn(outImage, kArgs.width, threadId, color, kArgs.numAccumulations, kArgs.useTonemapping, kArgs.tonemapWhitePoint);
}
////////////////////////////////////////////////////////