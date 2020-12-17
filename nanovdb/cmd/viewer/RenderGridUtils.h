// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderGridUtils.h

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
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int numAccumulations, const nanovdb::NanoGrid<ValueType>* grid, const SceneRenderParameters sceneParams, const MaterialParameters /*materialParams*/) const
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
            for (int sampleIndex = 0; sampleIndex < sceneParams.samplesPerPixel; ++sampleIndex) {
                uint32_t pixelSeed = hash((sampleIndex + (numAccumulations + 1) * sceneParams.samplesPerPixel)) ^ hash(ix, iy);

                RayT wRay = getRayFromPixelCoord(ix, iy, width, height, numAccumulations, sceneParams.samplesPerPixel, pixelSeed, sceneParams);
                RayT iRay = wRay.worldToIndexF(*grid);

                {
                    const auto& tree = grid->tree();
                    using TreeT = nanovdb::NanoTree<ValueType>;
                    auto acc = tree.getAccessor();
                    using AccT = decltype(acc);
                    nanovdb::TreeMarcher<typename TreeT::LeafNodeType, nanovdb::Ray<float>, AccT> marcher(acc);
                    if (marcher.init(iRay)) {
                        const typename TreeT::LeafNodeType* node = nullptr;
                        float                               t0 = 0, t1 = 0;
                        bool                                hitNode = marcher.step(&node, t0, t1);

                        if (!hitNode) {
                            auto envRadiance = traceEnvironment(wRay, sceneParams);
                            color[0] += envRadiance * 0.5f;
                            color[1] += envRadiance;
                            color[2] += envRadiance;
                        } else {
                            // color by a hashed value based on the node...
                            auto rgba = hash(reinterpret_cast<uintptr_t>(node));
                            color[0] += ((rgba >> 16) & 255) / 255.f;
                            color[1] += ((rgba >> 8) & 255) / 255.f;
                            color[2] += ((rgba >> 0) & 255) / 255.f;
                        }

                    } else {
                        auto envRadiance = traceEnvironment(wRay, sceneParams);

                        color[0] += envRadiance;
                        color[1] += envRadiance;
                        color[2] += envRadiance;
                    }
                }
            }

            for (int k = 0; k < 3; ++k)
                color[k] = color[k] / sceneParams.samplesPerPixel;
        }

        compositeFn(outPixel, color, numAccumulations, sceneParams);
    }
};

}
} // namespace render::grid