// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLevelSetUtils.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief General C++ implementation of the LevelSet rendering code.
*/

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>
#include "RenderConstants.h"
#include "RenderUtils.h"

namespace render {
namespace levelset {

struct RenderLevelSetRgba32fFn
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    template<typename ValueT>
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, Camera<float> camera, const nanovdb::NanoGrid<ValueT>* grid, int numAccumulations, const RenderConstants params) const
    {
        auto outPixel = &imgBuffer[4 * (ix + width * iy)];

        const Vec3T wLightDir = Vec3T(0, 1, 0);
        const Vec3T iLightDir = grid->worldToIndexDirF(wLightDir).normalize();

        const auto& tree = grid->tree();
        auto        acc = tree.getAccessor();

        float color[4] = {0, 0, 0, 0};

        for (int sampleIndex = 0; sampleIndex < params.samplesPerPixel; ++sampleIndex) {
            uint32_t pixelSeed = hash(sampleIndex + numAccumulations * params.samplesPerPixel ^ hash(ix, iy));

            float randVar1 = randomf(pixelSeed + 0);
            float randVar2 = randomf(pixelSeed + 1);

            float u = ix + 0.5f;
            float v = iy + 0.5f;

            if (numAccumulations > 0) {
#if 1
                float jitterX, jitterY;
                cmj(jitterX, jitterY, numAccumulations % 64, 8, 8, pixelSeed);
                u += jitterX - 0.5f;
                v += jitterY - 0.5f;
#else
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

            CoordT ijk;
            float  v0 = 0.0f;
            float  t;
            if (nanovdb::ZeroCrossing(iRay, acc, ijk, v0, t)) {
                Vec3T iPrimaryPos = Vec3T(RealT(ijk[0]), RealT(ijk[1]), RealT(ijk[2]));
                Vec3T wPrimaryPos = grid->indexToWorldF(iPrimaryPos);

                Vec3T iNormal(-v0);
                ijk[0] += 1;
                iNormal[0] += acc.getValue(ijk);
                ijk[0] -= 1;
                ijk[1] += 1;
                iNormal[1] += acc.getValue(ijk);
                ijk[1] -= 1;
                ijk[2] += 1;
                iNormal[2] += acc.getValue(ijk);
                iNormal.normalize();

                Vec3T iSurfacePos = iPrimaryPos + iNormal * 2.0f;

                Vec3T intensity = Vec3T(1);
                float occlusion = 0.0f;

                if (params.useOcclusion > 0) {
                    RayT iOccRay(iSurfacePos, lambertNoTangent(iNormal, randVar1, randVar2));
                    if (nanovdb::ZeroCrossing(iOccRay, acc, ijk, v0, t))
                        occlusion = params.useOcclusion;
                    intensity = Vec3T(1.0f - occlusion);
                }

                if (params.useLighting > 0) {
                    Vec3T diffuseKey = Vec3T(1);
                    Vec3T diffuseFill = Vec3T(1);
                    float ambient = 1.0f;
                    float specularKey = 0.f;
                    float shadowFactor = 0.0f;

                    Vec3T H = (iLightDir + -iRayDir).normalize();
                    specularKey = powf(fmaxf(0.0f, iNormal.dot(H)), 10.f);
                    const float diffuseWrap = 0.25f;
                    diffuseKey *= fmaxf(0.0f, (iNormal.dot(iLightDir) + diffuseWrap) / (1.0f + diffuseWrap));
                    diffuseFill *= fmaxf(0.0f, iNormal.dot(-iRayDir));

                    if (params.useShadows > 0) {
                        RayT iShadowRay(iSurfacePos, iLightDir);
                        if (nanovdb::ZeroCrossing(iShadowRay, acc, ijk, v0, t))
                            shadowFactor = params.useShadows;
                    }

                    intensity = params.useLighting * ((1.0f - shadowFactor) * (Vec3T(specularKey * 0.2f) + (diffuseKey * 0.8f)) + (1.0f - occlusion) * (diffuseFill * 0.2f + Vec3T(ambient * 0.1f)));
                    intensity = intensity + Vec3T((1.0f - params.useLighting) * (1.0f - occlusion));
                }

                color[0] += intensity[0];
                color[1] += intensity[1];
                color[2] += intensity[2];
            } else {
                float groundIntensity = 0.0f;
                float groundMix = 0.0f;

                if (params.useGround > 0) {
                    float wGroundT = (params.groundHeight - wRayEye[1]) / wRayDir[1];
                    if (wGroundT > 0.f) {
                        const Vec3T wGroundPos = wRayEye + wGroundT * wRayDir;
                        const Vec3T iGroundPos = grid->worldToIndexF(wGroundPos);
                        const Vec3T iGroundNormal = Vec3T(0,1,0);

                        rayTraceGround(wGroundT, params.groundFalloff, wGroundPos, wRayDir[1], groundIntensity, groundMix);

                        groundMix *= params.useGround;

                        if (params.useOcclusion > 0) {
                            RayT iOccRay(iGroundPos, lambertNoTangent(iGroundNormal, randVar1, randVar2));
                            if (nanovdb::ZeroCrossing(iOccRay, acc, ijk, v0, t))
                                groundIntensity = groundIntensity * (1.0f - params.useOcclusion);
                        }

                        if (params.useShadows > 0) {
                            RayT iShadowRay(iGroundPos, iLightDir);
                            if (nanovdb::ZeroCrossing(iShadowRay, acc, ijk, v0, t))
                                groundIntensity = groundIntensity - params.useShadows * groundIntensity;
                        }

                        if (params.useGroundReflections > 0) {
                            RayT iReflRay(iGroundPos, iRayDir - Vec3T(0.f, 2.0f * iRayDir[1], 0.f));
                            if (nanovdb::ZeroCrossing(iReflRay, acc, ijk, v0, t))
                                groundIntensity = groundIntensity - params.useGroundReflections * groundIntensity;
                        }
                    }
                }

                float skyIntensity = 0.75f + 0.25f * wRayDir[1];

                float bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;

                color[0] += bgIntensity;
                color[1] += bgIntensity;
                color[2] += bgIntensity;
            }
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
} // namespace render::levelset