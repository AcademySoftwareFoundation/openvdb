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
    float *h_outImage = reinterpret_cast<float*>(imageBuffer.data());
    RenderOp renderOp(handle, width, height);
    float sum = 0;
    if (auto *h_grid = handle.grid<float>()) {
        for (int i = 0; i < numIterations; ++i, sum += renderOp.renderImage(false, h_outImage, h_grid));
        std::cout << "Average of " << numIterations << " renderings (NanoVDB-Host) = " << (sum/numIterations) << " ms" << std::endl;
        renderOp.saveImage("raytrace_iso_surface-nanovdb-host.pfm", (float*)imageBuffer.data());

#if defined(NANOVDB_USE_CUDA)
        handle.deviceUpload();
        auto* d_grid = handle.deviceGrid<float>();
        if (!d_grid) throw std::runtime_error("GridHandle does not contain a valid device grid");
        imageBuffer.deviceUpload();
        float* d_outImage = reinterpret_cast<float*>(imageBuffer.deviceData());

        sum = 0;
        for (int i = 0; i < numIterations; ++i, sum += renderOp.renderImage(true, d_outImage, d_grid));
        std::cout << "Average of " << numIterations << " renderings (NanoVDB-Cuda) = " << (sum/numIterations) << " ms " << std::endl;
        imageBuffer.deviceDownload();
        renderOp.saveImage("raytrace_iso_surface-nanovdb-cuda.pfm", (float*)imageBuffer.data());
#endif
    } else if (auto *h_grid = handle.grid<nanovdb::ValueOnIndex>()) {
        for (int i = 0; i < numIterations; ++i, sum += renderOp.renderImage(false, h_outImage, h_grid));
        std::cout << "Average of " << numIterations << " renderings (NanoVDB-Host) = " << (sum/numIterations) << " ms" << std::endl;
        renderOp.saveImage("raytrace_iso_surface-nanovdb-host.pfm", (float*)imageBuffer.data());

#if defined(NANOVDB_USE_CUDA)
        handle.deviceUpload();
        auto* d_grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
        if (!d_grid) throw std::runtime_error("GridHandle does not contain a valid device grid");
        imageBuffer.deviceUpload();
        float* d_outImage = reinterpret_cast<float*>(imageBuffer.deviceData());

        sum = 0;
        for (int i = 0; i < numIterations; ++i, sum += renderOp.renderImage(true, d_outImage, d_grid));
        std::cout << "Average of " << numIterations << " renderings (NanoVDB-Cuda) = " << (sum/numIterations) << " ms " << std::endl;
        imageBuffer.deviceDownload();
        renderOp.saveImage("raytrace_iso_surface-nanovdb-cuda.pfm", (float*)imageBuffer.data());
#endif

    } else {
        throw std::runtime_error("GridHandle does not contain a valid device grid");
    }

}// runNanoVDB
