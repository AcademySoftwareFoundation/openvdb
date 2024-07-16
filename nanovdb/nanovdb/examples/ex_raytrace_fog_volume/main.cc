// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <algorithm>
#include <iostream>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreatePrimitives.h>

#if defined(NANOVDB_USE_CUDA)
#include <nanovdb/cuda/DeviceBuffer.h>
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

extern void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int width, int height, BufferT& imageBuffer);
#if defined(NANOVDB_USE_OPENVDB)
extern void runOpenVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int width, int height, BufferT& imageBuffer);
#endif

int main(int ac, char** av)
{
    try {
        nanovdb::GridHandle<BufferT> handle;
        if (ac > 1) {
            handle = nanovdb::io::readGrid<BufferT>(av[1]);
            std::cout << "Loaded NanoVDB grid[" << handle.gridMetaData()->shortGridName() << "]...\n";
        } else {
            handle = nanovdb::tools::createFogVolumeSphere<float, BufferT>(100.0f, nanovdb::Vec3d(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
        }

        if (handle.gridMetaData()->isFogVolume() == false) {
            throw std::runtime_error("Grid must be a fog volume");
        }

        const int numIterations = 50;

        const int width = 1024;
        const int height = 1024;
        BufferT   imageBuffer;
        imageBuffer.init(width * height * sizeof(float));

        runNanoVDB(handle, numIterations, width, height, imageBuffer);
#if defined(NANOVDB_USE_OPENVDB)
        runOpenVDB(handle, numIterations, width, height, imageBuffer);
#endif
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
