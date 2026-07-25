// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <iostream>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

extern void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int width, int height, BufferT& imageBuffer, bool usePersistentThreads);

int main(int ac, char** av)
{
    try {
        bool        usePersistentThreads = false;
        const char* gridName = nullptr;
        for (int i = 1; i < ac; ++i) {
            if (std::strcmp(av[i], "--persistent") == 0) {
                usePersistentThreads = true;
            } else if (!gridName) {
                gridName = av[i];
            } else {
                throw std::runtime_error("Usage: ex_raytrace_iso_surface [--persistent] [grid.nvdb]");
            }
        }
        nanovdb::GridHandle<BufferT> handle;
        if (gridName) {
            handle = nanovdb::io::readGrid<BufferT>(gridName);
            std::cout << "Loaded NanoVDB grid[" << handle.gridMetaData()->shortGridName() << "]...\n";
        } else {
            handle = nanovdb::tools::createLevelSetSphere<float, BufferT>(100.0f, nanovdb::Vec3d(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
        }

        const int numIterations = 50;
        const int width  = 4096;
        const int height = 4096;
        BufferT imageBuffer(width * height * sizeof(float));

        runNanoVDB(handle, numIterations, width, height, imageBuffer, usePersistentThreads);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
