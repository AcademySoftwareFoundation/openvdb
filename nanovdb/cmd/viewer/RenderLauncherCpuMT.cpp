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

template<typename FnT, typename... Args>
static void launchRender(int width, int height, const FnT& fn, Args... args)
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
}

bool RenderLauncherCpuMT::render(RenderMethod method, int width, int height, FrameBufferBase* imgBuffer, Camera<float> camera, const nanovdb::GridHandle<>& gridHdl, int numAccumulations, const RenderConstants& params, RenderStatistics* stats)
{
    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();
    
    float* imgPtr = (float*)imgBuffer->map((numAccumulations > 0) ? FrameBufferBase::AccessType::READ_WRITE : FrameBufferBase::AccessType::WRITE_ONLY);
    if (!imgPtr) {
        return false;
    }

    if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Float) {
        auto grid = gridHdl.grid<float>();

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        } else if (method == RenderMethod::FOG_VOLUME) {
            launchRender(width, height, render::fogvolume::RenderVolumeRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        } else if (method == RenderMethod::LEVELSET) {
            launchRender(width, height, render::levelset::RenderLevelSetRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Double) {
        auto grid = gridHdl.grid<double>();

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        } else if (method == RenderMethod::FOG_VOLUME) {
            launchRender(width, height, render::fogvolume::RenderVolumeRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        } else if (method == RenderMethod::LEVELSET) {
            launchRender(width, height, render::levelset::RenderLevelSetRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::UInt32) {
        auto grid = gridHdl.grid<uint32_t>();

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        } else if (method == RenderMethod::POINTS) {
            launchRender(width, height, render::points::RenderPointsRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Vec3f) {
        auto grid = gridHdl.grid<nanovdb::Vec3f>();

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        } else if (method == RenderMethod::FOG_VOLUME) {
            launchRender(width, height, render::fogvolume::RenderVolumeRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Vec3d) {
        auto grid = gridHdl.grid<nanovdb::Vec3d>();

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        } else if (method == RenderMethod::FOG_VOLUME) {
            launchRender(width, height, render::fogvolume::RenderVolumeRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Int64) {
        auto grid = gridHdl.grid<int64_t>();

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        } 
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Int32) {
        auto grid = gridHdl.grid<int32_t>();

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, grid, numAccumulations, params);
        } 
    } 

    imgBuffer->unmap();

    if (stats) {
        auto t1 = ClockT::now();
        stats->mDuration = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.f;
    }

    return true;
}
