// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>

#include <nanovdb/io/IO.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>

#include "common.h"

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

using namespace nanovdb;

template<typename UpdateOp>
float updateParticles(bool useCuda, const UpdateOp op, int numPoints, Vec3f* positions, Vec3f* velocities, const Grid<FloatTree>* grid)
{
    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    computeForEach(
        useCuda, numPoints, 512, __FILE__, __LINE__, [op, positions, velocities, grid] __hostdev__(int start, int end) {
            for (int i = start; i < end; ++i) {
                op(i, positions, velocities, grid);
            }
        });
    computeSync(useCuda, __FILE__, __LINE__);

    auto t1 = ClockT::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    return duration;
}

void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int numPoints, BufferT& positionBuffer, BufferT& velocityBuffer)
{
    using namespace nanovdb;

    auto* h_grid = handle.grid<float>();
    if (!h_grid) throw std::runtime_error("GridHandle does not contain a FloatGrid");

    Vec3f* h_positions = reinterpret_cast<Vec3f*>(positionBuffer.data());
    computeFill(false, h_positions, 0, sizeof(Vec3f) * numPoints);

    Vec3f* h_velocities = reinterpret_cast<Vec3f*>(velocityBuffer.data());
    computeFill(false, h_velocities, 0, sizeof(Vec3f) * numPoints);

    Vec3f wBBoxDim = Vec3f(h_grid->worldBBox().dim());
    Vec3f wBBoxBottomCenter = Vec3f(h_grid->worldBBox().min()) + wBBoxDim * Vec3f(0.5f, 0, 0.5f);

    const float dt = 0.01f;

    auto collisionResponse = [] __hostdev__(Vec3f pos, Vec3f nextPos, Vec3f normal, float d, Vec3f& outP, Vec3f& outV) {
        Vec3f delta = (nextPos - pos).normalize() * d;
        outP = nextPos + delta;
        outV = outV - 2.0f * (outV.dot(normal) * normal);
    };

    auto collideOp = [collisionResponse, dt] __hostdev__(const Grid<FloatTree>* grid, Vec3f wPos, Vec3f& wNextPos, Vec3f wVel, Vec3f& wNextVel) {
        // transform the position to grid index-space...
        Vec3f iNextPos = grid->worldToIndexF(wNextPos);
        // the grid index coordinate.
        auto ijk = Coord::Floor(iNextPos);
        // get an accessor.
        auto& tree = grid->tree();
        auto  acc = tree.getAccessor();
        if (tree.isActive(ijk)) { // are we inside the narrow band?
            auto wDistance = acc.getValue(ijk);
            if (wDistance <= 0) { // are we inside the levelset?
                // get the normal for collision resolution.
                Vec3f normal(-wDistance);
                ijk[0] += 1;
                normal[0] += acc.getValue(ijk);
                ijk[0] -= 1;
                ijk[1] += 1;
                normal[1] += acc.getValue(ijk);
                ijk[1] -= 1;
                ijk[2] += 1;
                normal[2] += acc.getValue(ijk);
                normal.normalize();

                // handle collision response with the surface.
                collisionResponse(wPos, wNextPos, normal, wDistance, wNextPos, wNextVel);
            }
        }
    };

    auto updateOp = [wBBoxBottomCenter, wBBoxDim, collideOp, dt] __hostdev__(int i, Vec3f* positions, Vec3f* velocities, const Grid<FloatTree>* grid) {
        Vec3f v = velocities[i];
        Vec3f p = positions[i];
        // gravity.
        Vec3f a = Vec3f(0, -9.8f, 0);
        // integrate.
        Vec3f nextP = p + v * dt + a * dt * dt;
        Vec3f nextV = v + a * dt;

        //printf("%f %f %f\n", p[0], p[1], p[2]);

        if (nextP[1] < 0) {
            // emit at top.
            nextP = Vec3f(randomf(i * 2 + 0), wBBoxDim[1], randomf(i * 2 + 1));
            nextV = Vec3f(0, 0, 0);
        } else {
            // collision response...
            collideOp(grid, p, nextP, v, nextV);
        }

        positions[i] = nextP;
        velocities[i] = nextV;
    };

    {
        float durationAvg = 0;
        for (int i = 0; i < numIterations; ++i) {
            float duration = updateParticles(false, updateOp, numPoints, h_positions, h_velocities, h_grid);
            //std::cout << "Duration(NanoVDB-Host) = " << duration << " ms" << std::endl;
            durationAvg += duration;
        }
        durationAvg /= numIterations;
        std::cout << "Average Duration(NanoVDB-Host) = " << durationAvg << " ms" << std::endl;
    }

#if defined(NANOVDB_USE_CUDA)

    handle.deviceUpload();

    auto* d_grid = handle.deviceGrid<float>();
    if (!d_grid)
        throw std::runtime_error("GridHandle does not contain a valid device grid");

    positionBuffer.deviceUpload();
    Vec3f* d_positions = reinterpret_cast<Vec3f*>(positionBuffer.deviceData());

    velocityBuffer.deviceUpload();
    Vec3f* d_velocities = reinterpret_cast<Vec3f*>(velocityBuffer.deviceData());

    {
        float durationAvg = 0;
        for (int i = 0; i < numIterations; ++i) {
            float duration = updateParticles(true, updateOp, numPoints, d_positions, d_velocities, d_grid);
            //std::cout << "Duration(NanoVDB-Cuda) = " << duration << " ms" << std::endl;
            durationAvg += duration;
        }
        durationAvg /= numIterations;
        std::cout << "Average Duration(NanoVDB-Cuda) = " << durationAvg << " ms" << std::endl;
    }
#endif
}
