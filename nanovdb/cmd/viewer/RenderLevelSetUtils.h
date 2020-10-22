// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLevelSetUtils.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief General C++ implementation of the LevelSet rendering code.
*/

#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>
#include "RenderConstants.h"
#include "RenderUtils.h"

namespace render {
namespace levelset {

template<typename ValueT, int InterpolationOrder>
struct RenderLevelSetRgba32fFn
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int numAccumulations, const nanovdb::BBoxR proxy, const nanovdb::NanoGrid<ValueT>* grid, const SceneRenderParameters sceneParams, const MaterialParameters params) const
    {
        auto outPixel = &imgBuffer[4 * (ix + width * iy)];

        float color[4] = {0, 0, 0, 0};

        if (!grid) {
            RayT wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);

            auto envRadiance = traceEnvironment(wRay, sceneParams);

            color[0] = envRadiance;
            color[1] = envRadiance;
            color[2] = envRadiance;

        } else {
            const Vec3T wLightDir = Vec3T(0, 1, 0);
            const Vec3T iLightDir = grid->worldToIndexDirF(wLightDir).normalize();

            const auto& tree = grid->tree();
            const auto  acc = tree.getAccessor();
            const auto  sampler = nanovdb::createSampler<0, decltype(acc), false>(acc);

            for (int sampleIndex = 0; sampleIndex < sceneParams.samplesPerPixel; ++sampleIndex) {
                uint32_t pixelSeed = hash((sampleIndex + (numAccumulations + 1) * sceneParams.samplesPerPixel)) ^ hash(ix, iy);

                RayT wRay = getRayFromPixelCoord(ix, iy, width, height, numAccumulations, sceneParams.samplesPerPixel, pixelSeed, sceneParams);

                RayT  iRay = wRay.worldToIndexF(*grid);

                CoordT ijk;
                ValueT v0 = 0.0f;
                float  t;
                if (nanovdb::ZeroCrossing(iRay, acc, ijk, v0, t)) {
                    Vec3T iPrimaryPos = Vec3T(RealT(ijk[0]), RealT(ijk[1]), RealT(ijk[2]));
                    Vec3T wPrimaryPos = grid->indexToWorldF(iPrimaryPos);

                    Vec3T iNormal((float)-v0);
                    ijk[0] += 1;
                    iNormal[0] += (float)acc.getValue(ijk);
                    ijk[0] -= 1;
                    ijk[1] += 1;
                    iNormal[1] += (float)acc.getValue(ijk);
                    ijk[1] -= 1;
                    ijk[2] += 1;
                    iNormal[2] += (float)acc.getValue(ijk);
                    iNormal.normalize();

                    Vec3T iSurfacePos = iPrimaryPos + iNormal * 2.0f;

                    Vec3T intensity = Vec3T(1);
                    float occlusion = 0.0f;

                    if (params.useOcclusion > 0) {
                        RayT iOccRay(iSurfacePos, lambertNoTangent(iNormal, randomXorShift(pixelSeed), randomXorShift(pixelSeed)));
                        if (nanovdb::ZeroCrossing(iOccRay, acc, ijk, v0, t))
                            occlusion = 1;
                        intensity = Vec3T(1.0f - occlusion);
                    }

                    if (sceneParams.useLighting) {
                        Vec3T diffuseKey = Vec3T(1);
                        Vec3T diffuseFill = Vec3T(1);
                        float ambient = 1.0f;
                        float specularKey = 0.f;
                        float shadowFactor = 0.0f;

                        Vec3T H = (iLightDir + -iRay.dir()).normalize();
                        specularKey = powf(fmaxf(0.0f, iNormal.dot(H)), 10.f);
                        const float diffuseWrap = 0.25f;
                        diffuseKey *= fmaxf(0.0f, (iNormal.dot(iLightDir) + diffuseWrap) / (1.0f + diffuseWrap));
                        diffuseFill *= fmaxf(0.0f, iNormal.dot(-iRay.dir()));

                        if (sceneParams.useShadows) {
                            RayT iShadowRay(iSurfacePos, iLightDir);
                            if (nanovdb::ZeroCrossing(iShadowRay, acc, ijk, v0, t))
                                shadowFactor = 1.f;
                        }

                        intensity = ((1.0f - shadowFactor) * (Vec3T(specularKey * 0.2f) + (diffuseKey * 0.8f)) + (1.0f - occlusion) * (diffuseFill * 0.2f + Vec3T(ambient * 0.1f)));
                    }

                    color[0] += intensity[0];
                    color[1] += intensity[1];
                    color[2] += intensity[2];
                } else {
                    float bgIntensity = 0;

                    if (sceneParams.useBackground) {
                        float groundIntensity = 0.0f;
                        float groundMix = 0.0f;

                        if (sceneParams.useGround) {
                            float wGroundT = (sceneParams.groundHeight - wRay.eye()[1]) / wRay.dir()[1];
                            if (wRay.dir()[1] != 0 && wGroundT > 0.f) {
                                const Vec3T wGroundPos = wRay(wGroundT);
                                const Vec3T iGroundPos = grid->worldToIndexF(wGroundPos);
                                const Vec3T iGroundNormal = Vec3T(0, 1, 0);

                                groundIntensity = evalGroundMaterial(wGroundT, sceneParams.groundFalloff, wGroundPos, wRay.dir()[1], groundMix);

                                if (params.useOcclusion) {
                                    RayT iOccRay(iGroundPos, lambertNoTangent(iGroundNormal, randomXorShift(pixelSeed), randomXorShift(pixelSeed)));
                                    if (nanovdb::ZeroCrossing(iOccRay, acc, ijk, v0, t))
                                        groundIntensity = 0.f;
                                }

                                if (sceneParams.useShadows) {
                                    RayT iShadowRay(iGroundPos, iLightDir);
                                    if (nanovdb::ZeroCrossing(iShadowRay, acc, ijk, v0, t))
                                        groundIntensity = 0.f;
                                }

                                if (sceneParams.useGroundReflections) {
                                    RayT iReflRay(iGroundPos, iRay.dir() - Vec3T(0.f, 2.0f * iRay.dir()[1], 0.f));
                                    if (nanovdb::ZeroCrossing(iReflRay, acc, ijk, v0, t))
                                        groundIntensity = 0;
                                }
                            }
                        }

                        float skyIntensity = evalSkyMaterial(wRay.dir());
                        bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
                    }

                    color[0] += bgIntensity;
                    color[1] += bgIntensity;
                    color[2] += bgIntensity;
                }
            }

            for (int k = 0; k < 3; ++k)
                color[k] = color[k] / sceneParams.samplesPerPixel;
        }

        compositeFn(outPixel, color, numAccumulations, sceneParams);
    }
};

}
} // namespace render::levelset