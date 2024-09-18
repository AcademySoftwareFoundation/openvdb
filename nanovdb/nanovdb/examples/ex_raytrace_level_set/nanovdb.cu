// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>

#if defined(NANOVDB_USE_CUDA)
#include <nanovdb/cuda/DeviceBuffer.h>
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif
#include <nanovdb/GridHandle.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>

#include "common.h"

void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int width, int height, BufferT& imageBuffer)
{
    using GridT = nanovdb::FloatGrid;
    using CoordT = nanovdb::Coord;
    using RealT = float;
    using Vec3T = nanovdb::math::Vec3<RealT>;
    using RayT = nanovdb::math::Ray<RealT>;

    auto *h_grid = handle.grid<float>();
    if (!h_grid)
        throw std::runtime_error("GridHandle does not contain a valid host grid");

    float* h_outImage = reinterpret_cast<float*>(imageBuffer.data());

    float              wBBoxDimZ = (float)h_grid->worldBBox().dim()[2] * 2;
    Vec3T              wBBoxCenter = Vec3T(h_grid->worldBBox().min() + h_grid->worldBBox().dim() * 0.5f);
    nanovdb::CoordBBox treeIndexBbox = h_grid->tree().bbox();
    std::cout << "Bounds: "
              << "[" << treeIndexBbox.min()[0] << "," << treeIndexBbox.min()[1] << "," << treeIndexBbox.min()[2] << "] -> ["
              << treeIndexBbox.max()[0] << "," << treeIndexBbox.max()[1] << "," << treeIndexBbox.max()[2] << "]" << std::endl;

    RayGenOp<Vec3T> rayGenOp(wBBoxDimZ, wBBoxCenter);
    CompositeOp     compositeOp;

    auto renderOp = [width, height, rayGenOp, compositeOp, treeIndexBbox, wBBoxDimZ] __hostdev__(int start, int end, float* image, const GridT* grid) {
        // get an accessor.
        auto acc = grid->tree().getAccessor();

        for (int i = start; i < end; ++i) {
            Vec3T rayEye;
            Vec3T rayDir;
            rayGenOp(i, width, height, rayEye, rayDir);
            // generate ray.
            RayT wRay(rayEye, rayDir);
            // transform the ray to the grid's index-space.
            RayT iRay = wRay.worldToIndexF(*grid);
            // intersect...
            float  t0;
            CoordT ijk;
            float  v;
            if (nanovdb::math::ZeroCrossing(iRay, acc, ijk, v, t0)) {
                // write distance to surface. (we assume it is a uniform voxel)
                float wT0 = t0 * float(grid->voxelSize()[0]);
                compositeOp(image, i, width, height, wT0 / (wBBoxDimZ * 2), 1.0f);
            } else {
                // write background value.
                compositeOp(image, i, width, height, 0.0f, 0.0f);
            }
        }
    };

    {
        float durationAvg = 0;
        for (int i = 0; i < numIterations; ++i) {
            float duration = renderImage(false, renderOp, width, height, h_outImage, h_grid);
            //std::cout << "Duration(NanoVDB-Host) = " << duration << " ms" << std::endl;
            durationAvg += duration;
        }
        durationAvg /= numIterations;
        std::cout << "Average Duration(NanoVDB-Host) = " << durationAvg << " ms" << std::endl;

        saveImage("raytrace_level_set-nanovdb-host.pfm", width, height, (float*)imageBuffer.data());
    }

#if defined(NANOVDB_USE_CUDA)
    handle.deviceUpload();

    auto* d_grid = handle.deviceGrid<float>();
    if (!d_grid)
        throw std::runtime_error("GridHandle does not contain a valid device grid");

    imageBuffer.deviceUpload();
    float* d_outImage = reinterpret_cast<float*>(imageBuffer.deviceData());

    {
        float durationAvg = 0;
        for (int i = 0; i < numIterations; ++i) {
            float duration = renderImage(true, renderOp, width, height, d_outImage, d_grid);
            //std::cout << "Duration(NanoVDB-Cuda) = " << duration << " ms" << std::endl;
            durationAvg += duration;
        }
        durationAvg /= numIterations;
        std::cout << "Average Duration(NanoVDB-Cuda) = " << durationAvg << " ms" << std::endl;

        imageBuffer.deviceDownload();
        saveImage("raytrace_level_set-nanovdb-cuda.pfm", width, height, (float*)imageBuffer.data());
    }
#endif
}
