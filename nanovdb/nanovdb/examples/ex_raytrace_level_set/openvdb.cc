// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#if defined(NANOVDB_USE_OPENVDB)

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>

#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb/tools/RayIntersector.h>

#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/NanoToOpenVDB.h>

#include "common.h"

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

void runOpenVDB(nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, int numIterations, int width, int height, BufferT& imageBuffer)
{
    using GridT = openvdb::FloatGrid;
    using CoordT = openvdb::Coord;
    using RealT = float;
    using Vec3T = openvdb::math::Vec3<RealT>;
    using RayT = openvdb::math::Ray<RealT>;

#if 1
    auto srcGrid = nanovdb::tools::nanoToOpenVDB(handle);
    std::cout << "Exporting to OpenVDB grid[" << srcGrid->getName() << "]...\n";
#else
    openvdb::initialize();
    std::string       filename = "C:/Users/william/Downloads/dragon.vdb";
    openvdb::io::File file(filename);
    file.open(false); //disable delayed loading
    auto srcGrid = file.readGrid<BufferT>("ls_dragon");
    std::cout << "Loading OpenVDB grid[" << srcGrid->getName() << "]...\n";
#endif

    auto h_grid = (GridT*)srcGrid.get();

    float* h_outImage = reinterpret_cast<float*>(imageBuffer.data());

    auto  indexBBox = h_grid->evalActiveVoxelBoundingBox();
    auto  gridXform = h_grid->transformPtr();
    auto  worldBBox = gridXform->indexToWorld(indexBBox);
    float wBBoxDimZ = (float)worldBBox.extents()[2] * 2;
    Vec3T wBBoxCenter = Vec3T(worldBBox.min() + worldBBox.extents() * 0.5f);

    RayGenOp<Vec3T> rayGenOp(wBBoxDimZ, wBBoxCenter);
    CompositeOp     compositeOp;

    openvdb::CoordBBox treeIndexBbox;
    treeIndexBbox = h_grid->evalActiveVoxelBoundingBox();
    std::cout << "Bounds: " << treeIndexBbox << std::endl;

    auto renderOp = [width, height, rayGenOp, compositeOp, treeIndexBbox, wBBoxDimZ] __hostdev__(int start, int end, float* image, const GridT* grid) {
        openvdb::tools::LevelSetRayIntersector<GridT, openvdb::tools::LinearSearchImpl<GridT, 0, RealT>, GridT::TreeType::RootNodeType::ChildNodeType::LEVEL, RayT> intersector(*grid);
        for (int i = start; i < end; ++i) {
            Vec3T rayEye;
            Vec3T rayDir;
            rayGenOp(i, width, height, rayEye, rayDir);
            // generate ray.
            RayT wRay(rayEye, rayDir);
            // transform the ray to the grid's index-space.
            RayT iRay = wRay.worldToIndex(*grid);
            // intersect...
            float t0;
            if (intersector.intersectsIS(iRay, t0)) {
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
            //std::cout << "Duration(OpenVDB-Host) = " << duration << " ms" << std::endl;
            durationAvg += duration;
        }
        durationAvg /= numIterations;
        std::cout << "Average Duration(OpenVDB-Host) = " << durationAvg << " ms" << std::endl;

        saveImage("raytrace_level_set-openvdb-host.pfm", width, height, (float*)imageBuffer.data());
    }
}

#endif