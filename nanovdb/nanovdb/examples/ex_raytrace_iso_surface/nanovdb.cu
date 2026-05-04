// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

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
    using CoordT = nanovdb::Coord;
    using Vec3T  = nanovdb::math::Vec3<float>;
    using RayT   = nanovdb::math::Ray<float>;

    const auto *metaData = handle.gridMetaData();
    const float dx = float(metaData->voxelSize()[0]), iso = dx;// <-- define iso-value as one voxel offset
    float wBBoxDimZ = (float)metaData->worldBBox().dim()[2] * 2;
    Vec3T wBBoxCenter = Vec3T(metaData->worldBBox().min() + metaData->worldBBox().dim() * 0.5f);
    RayGenOp<Vec3T> rayGenOp(wBBoxDimZ, wBBoxCenter);
    float *h_outImage = reinterpret_cast<float*>(imageBuffer.data());
    CompositeOp compositeOp;

    auto renderOp = [iso, width, height, rayGenOp, compositeOp, wBBoxDimZ] __hostdev__(int start, int end, float* image, const auto* grid) {
        auto acc = grid->tree().getAccessor();// get an accessor
        for (int i = start; i < end; ++i) {
            Vec3T rayEye, rayDir;
            rayGenOp(i, width, height, rayEye, rayDir);
            RayT wRay(rayEye, rayDir), iRay = wRay.worldToIndexF(*grid);// transform the ray to the grid's index-space.
            float  t0, v;
            CoordT ijk;
            if (nanovdb::math::isoCrossing(iRay, acc, ijk, v, t0, iso)) {// intersect...
                compositeOp(image, i, width, height, (t0 * dx) / (wBBoxDimZ * 2), 1.0f);
            } else {
                compositeOp(image, i, width, height, 0.0f, 0.0f);// write background value.
            }
        }
    };// renderOp lambda function

    float sum = 0;
    if (auto *h_grid = handle.grid<float>()) {
        for (int i = 0; i < numIterations; ++i, sum += renderImage(false, renderOp, width, height, h_outImage, h_grid));
        std::cout << "Average of " << numIterations << " renderings (NanoVDB-Host) = " << (sum/numIterations) << " ms" << std::endl;
        saveImage("raytrace_iso_surface-nanovdb-host.pfm", width, height, (float*)imageBuffer.data());

#if defined(NANOVDB_USE_CUDA)
        handle.deviceUpload();
        auto* d_grid = handle.deviceGrid<float>();
        if (!d_grid) throw std::runtime_error("GridHandle does not contain a valid device grid");
        imageBuffer.deviceUpload();
        float* d_outImage = reinterpret_cast<float*>(imageBuffer.deviceData());
        sum = 0;
        for (int i = 0; i < numIterations; ++i, sum += renderImage(true, renderOp, width, height, d_outImage, d_grid));
        std::cout << "Average of " << numIterations << " renderings (NanoVDB-Cuda) = " << (sum/numIterations) << " ms " << std::endl;
        imageBuffer.deviceDownload();
        saveImage("raytrace_iso_surface-nanovdb-cuda.pfm", width, height, (float*)imageBuffer.data());
#endif
    } else if () {

    }


}// runNanoVDB
