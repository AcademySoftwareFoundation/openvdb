// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLauncherCpuMT.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of CPU-multithreaded-platform Grid renderer.
*/

#include "RenderLauncherImpl.h"
#include "RenderFogVolumeUtils.h"
#include "RenderLevelSetUtils.h"
#include "RenderGridUtils.h"
#include "RenderPointsUtils.h"
#include "FrameBuffer.h"

#if defined(NANOVDB_USE_TBB)
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#else
#include <thread>
#include <atomic>
#endif
#include <iostream>

struct PlatformLauncherCpuMT
{
    template<typename ValueT>
    const nanovdb::NanoGrid<ValueT>* grid(const void* gridPtr) const
    {
        auto gridHdl = reinterpret_cast<const nanovdb::GridHandle<>*>(gridPtr);
        if (!gridHdl)
            return nullptr;
        return gridHdl->grid<ValueT>();        
    }

    template<typename FnT, typename... Args>
    bool render(int width, int height, const FnT& fn, Args... args) const
    {
#if defined(NANOVDB_USE_TBB)
        auto kernel2D = [&](const tbb::blocked_range2d<int>& r) {
            for (int iy = r.cols().begin(); iy < r.cols().end(); ++iy) {
                for (int ix = r.rows().begin(); ix < r.rows().end(); ++ix) {
                    fn(ix, iy, width, height, args...);
                }
            }
        };
        tbb::blocked_range2d<int> range2D(0, width, 0, height);
        tbb::parallel_for(range2D, kernel2D);
#else

        const int blockSize = 8;
        const int nBlocksX = (width + (blockSize - 1)) / blockSize;
        const int nBlocksY = (height + (blockSize - 1)) / blockSize;
        const int nBlocks = nBlocksX * nBlocksY;

        auto renderBlock = [=](int i) {
            const int blockOffsetX = blockSize * (i % nBlocksX);
            const int blockOffsetY = blockSize * (i / nBlocksY);

            for (int iy = 0; iy < blockSize; ++iy) {
                for (int ix = 0; ix < blockSize; ++ix) {
                    fn(ix + blockOffsetX, iy + blockOffsetY, width, height, args...);
                }
            }
        };

        unsigned                 numCores = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(numCores);
        std::atomic<int>         tileCounter(0);

        for (int i = 0; i < threads.size(); ++i) {
            threads[i] = std::thread([&tileCounter, nBlocks, i, renderBlock] {
                for (;;) {
                    const int blockIndex = tileCounter++;
                    if (blockIndex >= nBlocks) {
                        break;
                    }
                    renderBlock(blockIndex);
                }
            });
        }

        for (auto& t : threads)
            t.join();
#endif
        return true;
    }
};

bool RenderLauncherCpuMT::render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int /*numGrids*/, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats)
{
    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    float* imgPtr = (float*)imgBuffer->map((numAccumulations > 0) ? FrameBufferBase::AccessType::READ_WRITE : FrameBufferBase::AccessType::WRITE_ONLY);
    if (!imgPtr) {
        return false;
    }

    PlatformLauncherCpuMT methodLauncher;
    launchRender(methodLauncher, method, width, height, imgPtr, numAccumulations, grids, sceneParams, materialParams);

    imgBuffer->unmap();

    if (stats) {
        auto t1 = ClockT::now();
        stats->mDuration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    }

    return true;
}
