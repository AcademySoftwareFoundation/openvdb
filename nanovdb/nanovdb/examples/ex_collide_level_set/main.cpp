// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <algorithm>
#include <iostream>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::CudaDeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

extern void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int numPoints, BufferT& positionBuffer, BufferT& velocityBuffer);
#if defined(NANOVDB_USE_OPENVDB)
extern void runOpenVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int numPoints, BufferT& positionBuffer, BufferT& velocityBuffer);
#endif

int main(int ac, char** av)
{
    try {
        nanovdb::GridHandle<BufferT> handle;
        if (ac > 1) {
            handle = nanovdb::io::readGrid<BufferT>(av[1]);
            std::cout << "Loaded NanoVDB grid[" << handle.gridMetaData()->shortGridName() << "]...\n";
        } else {
            handle = nanovdb::createLevelSetSphere<float, float, BufferT>(100.0f, nanovdb::Vec3f(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3R(0), "sphere");
        }

        if (handle.gridMetaData()->isLevelSet() == false) {
            throw std::runtime_error("Grid must be a level set");
        }

        const int numIterations = 100;

        const int numPoints = 10000000;

        BufferT positionBuffer;
        positionBuffer.init(numPoints * sizeof(float) * 3);
        BufferT velocityBuffer;
        velocityBuffer.init(numPoints * sizeof(float) * 3);

        runNanoVDB(handle, numIterations, numPoints, positionBuffer, velocityBuffer);
#if defined(NANOVDB_USE_OPENVDB)
        runOpenVDB(handle, numIterations, numPoints, positionBuffer, velocityBuffer);
#endif
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
