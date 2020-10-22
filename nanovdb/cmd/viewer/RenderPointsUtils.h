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
namespace points {

struct PointCloud
{
    float density;
};

template<typename OutT, typename InT>
struct AttributeValueConverter
{
    inline __hostdev__ OutT operator()(const InT& in)
    {
        return OutT(in);
    }
};

template<>
struct AttributeValueConverter<float, nanovdb::Vec3f>
{
    inline __hostdev__ float operator()(const nanovdb::Vec3f& in)
    {
        return in[0];
    }
};

template<>
struct AttributeValueConverter<float, nanovdb::Vec3d>
{
    inline __hostdev__ float operator()(const nanovdb::Vec3d& in)
    {
        return float(in[0]);
    }
};

struct RenderPointsRgba32fFn
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    template<typename T, typename GridT>
    inline __hostdev__ static T sampleAttribute(const GridT& grid, CoordT origin, int i, const RendererAttributeParams& attribute)
    {
        if (attribute.attribute < 0)
            return T(attribute.offset);

        auto meta = grid.blindMetaData(attribute.attribute);
        auto data = grid.blindData(attribute.attribute);

        assert(i < meta.mElementCount);

        if (meta.mDataType == nanovdb::GridType::Vec3f)
            return AttributeValueConverter<T, nanovdb::Vec3f>()(*(reinterpret_cast<const nanovdb::Vec3f*>(data) + i)) * attribute.gain + T(attribute.offset);
        else if (meta.mDataType == nanovdb::GridType::Vec3d)
            return AttributeValueConverter<T, nanovdb::Vec3d>()(*(reinterpret_cast<const nanovdb::Vec3d*>(data) + i)) * attribute.gain + T(attribute.offset);
        else if (meta.mDataType == nanovdb::GridType::Float)
            return AttributeValueConverter<T, float>()(*(reinterpret_cast<const float*>(data) + i)) * attribute.gain + T(attribute.offset);

        return T(attribute.offset);
    }

    template<typename GridT, typename AccT>
    inline __hostdev__ static float getTransmittance(RayT& ray, const GridT& grid, AccT& acc, const PointCloud& geometry, uint32_t& seed, const nanovdb::Vec3f& lightDir, const MaterialParameters params)
    {
        using namespace nanovdb;

        float transmittance = 1.0f;

        const RendererAttributeParams radiusAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointRadius];
        const RendererAttributeParams positionAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointPosition];

        using TreeT = typename GridT::TreeType;
        nanovdb::TreeMarcher<typename TreeT::LeafNodeType, RayT, AccT> marcher(acc);
        if (marcher.init(ray)) {
            const typename TreeT::LeafNodeType* node;
            float                        t0 = 0, t1 = 0;

            while (marcher.step(&node, t0, t1)) {
                if (t0 >= t1)
                    continue;

                if (node->valueMax() == 0)
                    continue;
                auto nodeStartIndex = node->valueMin();

                // DDA through the node's values...
                nanovdb::HDDA<RayT> hdda;
                hdda.init(ray, t0, t1, 1);
                do {
                    auto           ijk = hdda.voxel();
                    const uint32_t offset = node->CoordToOffset(ijk);
                    if (node->isActive(offset)) {
                        auto beginIndex = nodeStartIndex + (offset == 0 ? 0 : node->getValue(offset - 1));
                        auto endIndex = nodeStartIndex + node->getValue(offset);

                        int count = endIndex - beginIndex;
                        for (int j = 0; j < count; ++j) {
                            int i = beginIndex + j;

                            const float radius = sampleAttribute<float>(grid, ijk, i, radiusAttribute);
                            const Vec3T pos = sampleAttribute<Vec3T>(grid, ijk, i, positionAttribute);

                            // distance to point from ray.
                            float s = ray.dir().cross(pos - ray.eye()).lengthSqr();
                            float f = s / (radius * radius);
                            if (f < 1.0f) {
                                f = geometry.density;

                                // modify transmittance.
                                transmittance *= 1.0f - f;
                            }
                        }
                    }
                } while (hdda.step());
            }
        }

        return transmittance;
    }

    template<typename GridT, typename AccT>
    inline __hostdev__ static Vec3T traceGeometry(RayT& ray, const GridT& grid, AccT& acc, const PointCloud& geometry, uint32_t& seed, const nanovdb::Vec3f& lightDir, float& transmittance, const MaterialParameters params)
    {
        using namespace nanovdb;

        const RendererAttributeParams radiusAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointRadius];
        const RendererAttributeParams positionAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointPosition];
        const RendererAttributeParams colorAttribute = params.attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointColor];

        Vec3T radiance(0);
        transmittance = 1.0f;

        using TreeT = typename GridT::TreeType;
        nanovdb::TreeMarcher<typename TreeT::LeafNodeType, RayT, AccT> marcher(acc);
        if (marcher.init(ray)) {
            const typename TreeT::LeafNodeType* node;
            float                        t0 = 0, t1 = 0;

            while (marcher.step(&node, t0, t1)) {
                if (t0 >= t1)
                    continue;

                if (node->valueMax() == 0)
                    continue;
                auto nodeStartIndex = node->valueMin();

                // DDA through the node's values...
                nanovdb::HDDA<RayT> hdda;
                hdda.init(ray, t0, t1, 1);
                do {
                    auto           ijk = hdda.voxel();
                    const uint32_t offset = node->CoordToOffset(ijk);
                    if (node->isActive(offset)) {
                        auto beginIndex = nodeStartIndex + (offset == 0 ? 0 : node->getValue(offset - 1));
                        auto endIndex = nodeStartIndex + node->getValue(offset);

                        int count = endIndex - beginIndex;
                        for (int j = 0; j < count; ++j) {
                            int i = beginIndex + j;

                            const float radius = sampleAttribute<float>(grid, ijk, i, radiusAttribute);
                            const Vec3T pos = sampleAttribute<Vec3T>(grid, ijk, i, positionAttribute);

                            // distance to point from ray.
                            float s = ray.dir().cross(pos - ray.eye()).lengthSqr();
                            float f = s / (radius * radius);
                            if (f < 1.0f) {
                                f = geometry.density;

                                float shadowTransmittance = 1.0f;
#if 0
								if (sceneParams.useShadows)
								{
									RayT iShadowRay(pos + lightDir * 0.1f, lightDir);
									shadowTransmittance = getTransmittance(iShadowRay, grid, acc, geometry, seed, lightDir);
								}
#endif

                                radiance += shadowTransmittance * transmittance * f * sampleAttribute<Vec3T>(grid, ijk, i, colorAttribute);

                                // modify transmittance.
                                transmittance *= 1.0f - f;
                            }
                        }
                    }
                } while (hdda.step());
            }
        }

        return radiance;
    }

    template<typename ValueType>
    inline __hostdev__ void operator()(int ix, int iy, int width, int height, float* imgBuffer, int numAccumulations, const nanovdb::NanoGrid<ValueType>* grid, const SceneRenderParameters sceneParams, const MaterialParameters params) const
    {
        float* outPixel = &imgBuffer[4 * (ix + width * iy)];

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

            PointCloud geometry;
            geometry.density = params.volumeDensityScale;

            for (int sampleIndex = 0; sampleIndex < sceneParams.samplesPerPixel; ++sampleIndex) {
                uint32_t pixelSeed = hash((sampleIndex + (numAccumulations + 1) * sceneParams.samplesPerPixel)) ^ hash(ix, iy);

                RayT wRay = getRayFromPixelCoord(ix, iy, width, height, numAccumulations, sceneParams.samplesPerPixel, pixelSeed, sceneParams);

                RayT  iRay = wRay.worldToIndexF(*grid);
                Vec3T iRayDir = iRay.dir();

                float pathThroughput = 1.0f;
                Vec3T radiance = traceGeometry(iRay, *grid, acc, geometry, pixelSeed, iLightDir, pathThroughput, params);

                if (pathThroughput > 0.0f) {
                    float bgIntensity = 0;

                    if (sceneParams.useBackground) {
                        float groundIntensity = 0.0f;
                        float groundMix = 0.0f;

                        if (sceneParams.useLighting && sceneParams.useGround) {
                            // intersect with ground plane and draw checker if camera is above...
                            float wGroundT = (sceneParams.groundHeight - wRay.eye()[1]) / wRay.dir()[1];
                            if (wRay.dir()[1] != 0 && wGroundT > 0.f) {
                                Vec3T wGroundPos = wRay(wGroundT);
                                Vec3T iGroundPos = grid->worldToIndexF(wGroundPos);

                                groundIntensity = evalGroundMaterial(wGroundT, sceneParams.groundFalloff, wGroundPos, wRay.dir()[1], groundMix);

                                if (sceneParams.useShadows) {
                                    RayT  iShadowRay(iGroundPos, iLightDir);
                                    float shadowTransmittance = getTransmittance(iShadowRay, *grid, acc, geometry, pixelSeed, iLightDir, params);
                                    groundIntensity *= shadowTransmittance;
                                }
                            }
                        }
                        float skyIntensity = evalSkyMaterial(wRay.dir());
                        bgIntensity = (1.f - groundMix) * skyIntensity + groundMix * groundIntensity;
                    }
                    radiance += nanovdb::Vec3f(pathThroughput * bgIntensity);
                }

                color[0] += radiance[0];
                color[1] += radiance[1];
                color[2] += radiance[2];
            }

            for (int k = 0; k < 3; ++k)
                color[k] = color[k] / sceneParams.samplesPerPixel;
        }

        compositeFn(outPixel, color, numAccumulations, sceneParams);
    }
};

}
} // namespace render::points