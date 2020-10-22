// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderFunctions.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief General C++ implementation of the Grid rendering code.
*/

#pragma once

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
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int numAccumulations, const nanovdb::NanoGrid<ValueType>* grid, const SceneRenderParameters sceneParams, const MaterialParameters params) const
    {
        int pixelSeed = numAccumulations + hash(ix, iy);

        auto outPixel = &imgBuffer[4 * (ix + width * iy)];

        float color[4] = {0, 0, 0, 0};

        if (!grid) {
            RayT wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);

            auto envRadiance = traceEnvironment(wRay, sceneParams);

            color[0] = envRadiance;
            color[1] = envRadiance;
            color[2] = envRadiance;
        } else {
            
            float groundIntensity = 0.0f;
            float groundMix = 0.0f;

            RayT wRay = getRayFromPixelCoord(ix, iy, width, height, sceneParams);

            RayT  iRay = wRay.worldToIndexF(*grid);
            Vec3T iRayDir = iRay.dir();
            Vec3T wRayDir = wRay.dir();
            Vec3T wRayEye = wRay.eye();

            {
                const auto& tree = grid->tree();
                using TreeT = nanovdb::NanoTree<ValueType>;
                auto acc = tree.getAccessor();
                using AccT = decltype(acc);
                nanovdb::TreeMarcher<typename TreeT::LeafNodeType, nanovdb::Ray<float>, AccT> marcher(acc);
                if (marcher.init(iRay)) {
                    float                        nodeSpanAccum = 0;
                    float                        nodeCount = 0;
                    const typename TreeT::LeafNodeType* node;
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
                    auto envRadiance = traceEnvironment(wRay, sceneParams);

                    color[0] = envRadiance;
                    color[1] = envRadiance;
                    color[2] = envRadiance;
                }
            }
        }

        for (int k = 0; k < 3; ++k)
            outPixel[k] = color[k];
        outPixel[3] = 1.0;
    }
};
}
} // namespace render::grid