// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>

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

// Number of untimed warmup iterations to run before the timed loop. These
// pay the kernel cold-start, lazy module load and GPU clock ramp-up costs
// that would otherwise contaminate the first measured iteration.
static constexpr int NUM_WARMUP_ITERATIONS = 2;

// Print min / median / mean / max instead of just mean. A single outlier
// (context switch, OS jitter, thermal blip) pulls the mean noticeably but
// leaves the median alone, so the median is the more trustworthy stat.
static void
reportStats(const char *label, const std::vector<float> &samples)
{
    if (samples.empty()) return;
    std::vector<float> sorted(samples); // copy so we don't reorder the caller's data
    std::sort(sorted.begin(), sorted.end());
    const float minMs = sorted.front();
    const float maxMs = sorted.back();
    const float medianMs = sorted[sorted.size() / 2];
    float sum = 0;
    for (float s : sorted) sum += s;
    const float meanMs = sum / float(sorted.size());
    std::cout << label
              << " min=" << minMs << " ms"
              << "  median=" << medianMs << " ms"
              << "  mean=" << meanMs << " ms"
              << "  max=" << maxMs << " ms"
              << "  (n=" << sorted.size() << ")" << std::endl;
}

void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int width, int height, BufferT& imageBuffer)
{
    using GridT  = nanovdb::FloatGrid;
    using CoordT = nanovdb::Coord;
    using RealT  = float;
    using Vec3T  = nanovdb::math::Vec3<RealT>;
    using RayT   = nanovdb::math::Ray<RealT>;

    auto *h_grid = handle.grid<float>();
    if (!h_grid) throw std::runtime_error("GridHandle does not contain a valid host grid");

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
            if (nanovdb::math::zeroCrossing(iRay, acc, ijk, v, t0)) {
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
        for (int i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
            (void)renderImage(false, renderOp, width, height, h_outImage, h_grid);
        }

        std::vector<float> samples;
        samples.reserve(numIterations);
        for (int i = 0; i < numIterations; ++i) {
            samples.push_back(renderImage(false, renderOp, width, height, h_outImage, h_grid));
        }
        reportStats("Duration(NanoVDB-Host):", samples);

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
        for (int i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
            (void)renderImage(true, renderOp, width, height, d_outImage, d_grid);
        }

        std::vector<float> samples;
        samples.reserve(numIterations);
        for (int i = 0; i < numIterations; ++i) {
            samples.push_back(renderImage(true, renderOp, width, height, d_outImage, d_grid));
        }
        reportStats("Duration(NanoVDB-Cuda):", samples);

        imageBuffer.deviceDownload();
        saveImage("raytrace_level_set-nanovdb-cuda.pfm", width, height, (float*)imageBuffer.data());
    }
#endif
}
