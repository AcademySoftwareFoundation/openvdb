// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderFunctions.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief General C++ implementation of the Grid rendering code.
*/

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>
#include "RenderConstants.h"
#include "RenderUtils.h"

namespace render {
namespace grid {

struct RenderGridRgba32fFn
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    template<typename ValueType>
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, Camera<float> camera, const nanovdb::NanoGrid<ValueType>* grid, int numAccumulations, const RenderConstants params) const
    {
        int pixelSeed = numAccumulations + hash(ix, iy);

        auto outPixel = &imgBuffer[4 * (ix + width * iy)];

        const auto& tree = grid->tree();

        float u = ix + 0.5f;
        float v = iy + 0.5f;

        u /= width;
        v /= height;

        RayT wRay = camera.getRay(u, v);
        RayT iRay = wRay.worldToIndexF(*grid);

        Vec3T iRayDir = iRay.dir();
        Vec3T wRayDir = wRay.dir();
        Vec3T wRayEye = wRay.eye();

        float color[4];

        {
            auto acc = tree.getAccessor();
            using AccT = decltype(acc);
            nanovdb::TreeMarcher<typename AccT::NodeT0, nanovdb::Ray<float>, AccT> marcher(acc);
            if (marcher.init(iRay)) {
                float                        nodeSpanAccum = 0;
                float                        nodeCount = 0;
                const typename AccT::NodeT0* node;
                float                        t0 = 0, t1 = 0;
                while (marcher.step(&node, t0, t1)) {
                    nodeSpanAccum += (t1 - t0);
                    nodeCount += 1.0f;
                }

                if (nodeCount == 0) {
                    // hit no nodes!
                    color[0] = 0;
                    color[1] = 1;
                    color[2] = 0;
                } else {
                    color[0] = nodeCount / 16.f;
                    color[1] = 0;
                    color[2] = 0;
                }
            } else {
                float groundIntensity = 0.0f;
                float groundMix = 0.0f;

                if (params.useGround > 0) {
                    float wGroundT = (params.groundHeight - wRayEye[1]) / wRayDir[1];
                    if (wGroundT > 0.f) {
                        Vec3T wGroundPos = wRayEye + wGroundT * wRayDir;
                        Vec3T iGroundPos = grid->worldToIndexF(wGroundPos);

                        rayTraceGround(wGroundT, params.groundFalloff, wGroundPos, wRayDir[1], groundIntensity, groundMix);

                        groundMix *= params.useGround;
                    }
                }

                float skyIntensity = 0.75f + 0.25f * wRayDir[1];

                float bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;

                color[0] = bgIntensity;
                color[1] = bgIntensity;
                color[2] = bgIntensity;
            }
        }

        for (int k = 0; k < 3; ++k)
            outPixel[k] = color[k];
        outPixel[3] = 1.0;
    }
};

}
} // namespace render::grid