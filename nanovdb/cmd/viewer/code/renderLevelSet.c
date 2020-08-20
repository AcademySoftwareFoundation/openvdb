// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

//	\file renderLevelSet.c
//	\author Wil Braithwaite
//	\date July 24, 2020
//	\brief General (C99-ish) implementation of the LevelSet rendering code.
//
////////////////////////////////////////////////////////

#if defined(CNANOVDB_COMPILER_GLSL)

layout(rgba32f, binding = 0) uniform image2D outImage;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    int cxt;

    ivec2 threadId = getThreadId();

#elif defined(__OPENCL_VERSION__)

CNANOVDB_KERNEL void renderLevelSet(
    CNANOVDB_GLOBAL vec4* outImage,
    CNANOVDB_GLOBAL const nanovdb_Node0_float* nodeLevel0,
    CNANOVDB_GLOBAL const nanovdb_Node1_float* nodeLevel1,
    CNANOVDB_GLOBAL const nanovdb_Node2_float* nodeLevel2,
    CNANOVDB_GLOBAL const nanovdb_RootData_float* rootData,
    CNANOVDB_GLOBAL const nanovdb_RootData_Tile_float* rootDataTiles,
    CNANOVDB_GLOBAL const nanovdb_GridData* gridData,
    ArgUniforms                             kArgs)
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

CNANOVDB_KERNEL void renderLevelSet(
    ivec2           threadId,
    CNANOVDB_GLOBAL vec4* outImage,
    CNANOVDB_GLOBAL const nanovdb_Node0_float* nodeLevel0,
    CNANOVDB_GLOBAL const nanovdb_Node1_float* nodeLevel1,
    CNANOVDB_GLOBAL const nanovdb_Node2_float* nodeLevel2,
    CNANOVDB_GLOBAL const nanovdb_RootData_float* rootData,
    CNANOVDB_GLOBAL const nanovdb_RootData_Tile_float* rootDataTiles,
    CNANOVDB_GLOBAL const nanovdb_GridData* gridData,
    ArgUniforms                             kArgs)
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

    uint32_t pixelSeed = kArgs.numAccumulations + hash2(ix, iy);

    vec3 cameraP = CNANOVDB_MAKE_VEC3(kArgs.cameraPx, kArgs.cameraPy, kArgs.cameraPz);
    vec3 cameraU = CNANOVDB_MAKE_VEC3(kArgs.cameraUx, kArgs.cameraUy, kArgs.cameraUz);
    vec3 cameraV = CNANOVDB_MAKE_VEC3(kArgs.cameraVx, kArgs.cameraVy, kArgs.cameraVz);
    vec3 cameraW = CNANOVDB_MAKE_VEC3(kArgs.cameraWx, kArgs.cameraWy, kArgs.cameraWz);

    const vec3 wLightDir = CNANOVDB_MAKE_VEC3(0.0f, 1.0f, 0.0f);
    const vec3 iLightDir = vec3_normalize(nanovdb_Grid_worldToIndexDirF(CNANOVDB_GRIDDATA(cxt).grid, wLightDir));

#if 0
    {
        vec4 color = CNANOVDB_MAKE_VEC4(fabs(iRayDir.x), fabs(iRayDir.y), fabs(iRayDir.z), 1);
        imageStorePixel(outImage, kArgs.width, threadId, color);
        return;
    }
#endif

    nanovdb_ReadAccessor acc = nanovdb_ReadAccessor_create();
    vec3                 color = CNANOVDB_MAKE_VEC3(0, 0, 0);

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

        nanovdb_Coord ijk;
        float         v0 = 0.0f;
        if (nanovdb_ZeroCrossing(cxt, iRay, acc, CNANOVDB_ADDRESS(ijk), CNANOVDB_ADDRESS(v0))) {
#if 0
		// DEBUG...
		color = CNANOVDB_MAKE_VEC4(v0,0,0,1);
		imageStorePixel(outImage, kArgs.width, threadId, color);
		return;
#endif

            Vec3T iPrimaryPos = nanovdb_CoordToVec3f(ijk);
            //Vec3T wPrimaryPos = nanovdb_Grid_indexToWorldF(grid, iPrimaryPos);

            vec3 iNormal = makeVec3(-v0);
            ijk.mVec[0] += 1;
            iNormal.x += nanovdb_ReadAccessor_getValue(cxt, CNANOVDB_ADDRESS(acc), ijk);
            ijk.mVec[0] -= 1;
            ijk.mVec[1] += 1;
            iNormal.y += nanovdb_ReadAccessor_getValue(cxt, CNANOVDB_ADDRESS(acc), ijk);
            ijk.mVec[1] -= 1;
            ijk.mVec[2] += 1;
            iNormal.z += nanovdb_ReadAccessor_getValue(cxt, CNANOVDB_ADDRESS(acc), ijk);
            iNormal = vec3_normalize(iNormal);

            Vec3T intensity = makeVec3(1);
            float occlusion = 0.0f;

            if (kArgs.useOcclusion > 0) {
                nanovdb_Ray iOccRay;
                iOccRay.mEye = vec3_add(iPrimaryPos, vec3_fmul(2.0f, iNormal));
                iOccRay.mDir = lambertNoTangent(iNormal, randVar1, randVar2);
                iOccRay.mT0 = DeltaFloat;
                iOccRay.mT1 = MaxFloat;
                if (nanovdb_ZeroCrossing(cxt, iOccRay, acc, CNANOVDB_ADDRESS(ijk), CNANOVDB_ADDRESS(v0)))
                    occlusion = kArgs.useOcclusion;
                intensity = makeVec3(1.0f - occlusion);
            }

            if (kArgs.useLighting > 0) {
                Vec3T diffuseKey = makeVec3(1);
                Vec3T diffuseFill = makeVec3(1);
                float ambient = 1.0f;
                float specularKey = 0.f;
                float shadowFactor = 0.0f;

                Vec3T H = vec3_normalize(vec3_sub(iLightDir, iRayDir));
                specularKey = pow(fmax(0.0f, vec3_dot(iNormal, H)), 10.f);
                const float diffuseWrap = 0.25f;
                diffuseKey = vec3_fmul(fmax(0.0f, (vec3_dot(iNormal, iLightDir) + diffuseWrap) / (1.0f + diffuseWrap)), diffuseKey);
                diffuseFill = vec3_fmul(fmax(0.0f, -vec3_dot(iNormal, iRayDir)), diffuseFill);

                if (kArgs.useShadows > 0) {
                    nanovdb_Ray iShadowRay;
                    iShadowRay.mEye = vec3_add(iPrimaryPos, vec3_fmul(2.0f, iNormal));
                    iShadowRay.mDir = iLightDir;
                    iShadowRay.mT0 = DeltaFloat;
                    iShadowRay.mT1 = MaxFloat;
                    if (nanovdb_ZeroCrossing(cxt, iShadowRay, acc, CNANOVDB_ADDRESS(ijk), CNANOVDB_ADDRESS(v0)))
                        shadowFactor = kArgs.useShadows;
                }

                intensity = vec3_fmul((1.0f - shadowFactor), vec3_add(makeVec3(specularKey * 0.2f), vec3_fmul(0.8f, diffuseKey)));
                intensity = vec3_add(intensity, vec3_fmul((1.0f - occlusion), vec3_add(vec3_fmul(0.2f, diffuseFill), makeVec3(ambient * 0.1f))));
                intensity = vec3_fmul(kArgs.useLighting, intensity);
                intensity = vec3_add(intensity, makeVec3((1.0f - kArgs.useLighting) * (1.0f - occlusion)));
            }

            color.x += intensity.x;
            color.y += intensity.y;
            color.z += intensity.z;
        } else {
            float groundIntensity = 0.0f;
            float groundMix = 0.0f;

            if (kArgs.useGround > 0) {
                float wGroundT = (kArgs.groundHeight - wRayEye.y) / wRayDir.y;
                if (wGroundT > 0.f) {
                    vec3 wGroundPos = vec3_add(wRayEye, vec3_fmul(wGroundT, wRayDir));
                    vec3 iGroundPos = nanovdb_Grid_worldToIndexF(CNANOVDB_GRIDDATA(cxt).grid, wGroundPos);

                    rayTraceGround(wGroundT, kArgs.groundFalloff, wGroundPos, wRayDir.y, CNANOVDB_ADDRESS(groundIntensity), CNANOVDB_ADDRESS(groundMix));

                    groundMix *= kArgs.useGround;

                    if (kArgs.useShadows > 0) {
                        nanovdb_Ray iShadowRay;
                        iShadowRay.mEye = iGroundPos;
                        iShadowRay.mDir = iLightDir;
                        iShadowRay.mT0 = DeltaFloat;
                        iShadowRay.mT1 = MaxFloat;
                        if (nanovdb_ZeroCrossing(cxt, iShadowRay, acc, CNANOVDB_ADDRESS(ijk), CNANOVDB_ADDRESS(v0)))
                            groundIntensity = groundIntensity - kArgs.useShadows * groundIntensity;
                    }

                    if (kArgs.useGroundReflections > 0) {
                        nanovdb_Ray iReflRay;
                        iReflRay.mEye = iGroundPos;
                        iReflRay.mDir = vec3_sub(iRayDir, CNANOVDB_MAKE_VEC3(0.f, 2.0f * iRayDir.y, 0.f));
                        iReflRay.mT0 = DeltaFloat;
                        iReflRay.mT1 = MaxFloat;
                        if (nanovdb_ZeroCrossing(cxt, iReflRay, acc, CNANOVDB_ADDRESS(ijk), CNANOVDB_ADDRESS(v0)))
                            groundIntensity = groundIntensity - kArgs.useGroundReflections * groundIntensity;
                    }
                }
            }

            float skyIntensity = 0.75f + 0.25f * wRayDir.y;
            float bgIntensity = (1.0f - groundMix) * skyIntensity + groundMix * groundIntensity;

            color.x += bgIntensity;
            color.y += bgIntensity;
            color.z += bgIntensity;
        }
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