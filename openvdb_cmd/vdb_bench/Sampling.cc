// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file Sampling.cc
///
/// @brief Group sampling: resampling, multi resolution grids, volume advection and
/// sampling onto points.

#include "Harness.h"

#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointSample.h>
#include <openvdb/points/PointScatter.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/MultiResGrid.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeAdvect.h>

#include <memory>

namespace bench {

namespace {

// The cost of these tools grows with the volume, so they run on a coarser grid than the rest
const float kSamplingVoxelSize = 1.0f / 256.0f;

template<typename SamplerT>
void
resampleRow(Context& context, const std::string& samplerName, const FloatGrid& source)
{
    const std::string tool = "resampleToMatch";
    const double voxelSize = source.voxelSize()[0];

    struct Target { std::string name; math::Transform::Ptr transform; };
    std::vector<Target> targets;
    targets.push_back({"scale 0.5", math::Transform::createLinearTransform(voxelSize * 2.0)});
    {
        math::Transform::Ptr offset = math::Transform::createLinearTransform(voxelSize);
        offset->postTranslate(Vec3d(0.5 * voxelSize));
        targets.push_back({"scale 1 offset half a voxel", offset});
    }
    targets.push_back({"scale 2", math::Transform::createLinearTransform(voxelSize * 0.5)});
    {
        math::Transform::Ptr rotated = math::Transform::createLinearTransform(voxelSize);
        rotated->postRotate(math::pi<double>() / 6.0, math::Y_AXIS);
        targets.push_back({"rotation 30 degrees", rotated});
    }

    for (const Target& target : targets) {
        const std::string name = samplerName + " " + target.name;
        if (!context.wanted(tool, name)) continue;
        context.progress(tool, name);
        FloatGrid::Ptr last;
        const Timing timing = context.time(
            [&]() {
                FloatGrid::Ptr destination = FloatGrid::create(source.background());
                destination->setTransform(target.transform->copy());
                destination->setGridClass(source.getGridClass());
                return destination;
            },
            [&](FloatGrid::Ptr& destination) {
                tools::resampleToMatch<SamplerT>(source, *destination);
                last = destination;
            });
        // Tasks that overlap write the output in no fixed order, so the exact values vary
        Digest digest;
        digest.addGridCoarse(*last);
        context.record(tool, name, timing, digest);
    }
}

void
runResample(Context& context)
{
    FloatGrid::Ptr fog = makeFogSphere(kSamplingVoxelSize);
    resampleRow<tools::PointSampler>(context, "PointSampler", *fog);
    resampleRow<tools::BoxSampler>(context, "BoxSampler", *fog);
    resampleRow<tools::QuadraticSampler>(context, "QuadraticSampler", *fog);
}

void
runMultiRes(Context& context)
{
    using MultiRes = tools::MultiResGrid<FloatTree>;
    FloatGrid::Ptr sphere = makeSphere<FloatGrid>(kSamplingVoxelSize);
    const std::string tool = "MultiResGrid";
    const size_t levels = 4;

    auto addLevels = [&](Digest& digest, MultiRes& multiRes) {
        for (size_t level = 0; level < multiRes.numLevels(); ++level) {
            digest.addGrid(*multiRes.grid(level));
        }
    };

    {
        const std::string name = "construct 4 levels (restriction)";
        if (context.wanted(tool, name)) {
            context.progress(tool, name);
            std::unique_ptr<MultiRes> multiRes;
            const Timing timing = context.time(
                []() { return 0; },
                [&](int&) { multiRes.reset(new MultiRes(levels, *sphere)); });
            Digest digest;
            addLevels(digest, *multiRes);
            context.record(tool, name, timing, digest);
        }
    }
    for (size_t destination = 0; destination < 2; ++destination) {
        const std::string name = "prolongateActiveVoxels level " + std::to_string(destination);
        if (!context.wanted(tool, name)) continue;
        context.progress(tool, name);
        std::unique_ptr<MultiRes> multiRes;
        const Timing timing = context.time(
            [&]() { return std::unique_ptr<MultiRes>(new MultiRes(levels, *sphere)); },
            [&](std::unique_ptr<MultiRes>& state) {
                state->prolongateActiveVoxels(destination);
                multiRes = std::move(state);
            });
        Digest digest;
        addLevels(digest, *multiRes);
        context.record(tool, name, timing, digest);
    }
    {
        const std::string name = "createGrid 1.5 order 1";
        if (context.wanted(tool, name)) {
            context.progress(tool, name);
            MultiRes multiRes(levels, *sphere);
            FloatGrid::Ptr result;
            const Timing timing = context.time(
                []() { return 0; },
                [&](int&) { result = multiRes.createGrid<1>(1.5f); });
            Digest digest;
            digest.addGrid(*result);
            context.record(tool, name, timing, digest);
        }
    }
}

void
runVolumeAdvection(Context& context)
{
    const float voxelSize = kSamplingVoxelSize;
    FloatGrid::Ptr fog = makeFogSphere(voxelSize);
    Vec3SGrid::Ptr velocity = makeVelocityGrid();

    struct Integrator { const char* name; tools::Scheme::SemiLagrangian scheme; };
    const Integrator integrators[] = {
        {"SEMI", tools::Scheme::SEMI}, {"RK4", tools::Scheme::RK4},
        {"MAC", tools::Scheme::MAC}, {"BFECC", tools::Scheme::BFECC}};

    const std::string tool = "VolumeAdvection";
    for (const Integrator& integrator : integrators) {
        for (double courant : {0.5, 4.0}) {
            const std::string name = std::string(integrator.name) + " CFL " + (courant < 1.0 ? "0.5" : "4");
            if (!context.wanted(tool, name)) continue;
            context.progress(tool, name);
            FloatGrid::Ptr last;
            const Timing timing = context.time(
                []() { return 0; },
                [&](int&) {
                    tools::VolumeAdvection<Vec3SGrid> advection(*velocity);
                    advection.setIntegrator(integrator.scheme);
                    const double dt = courant * double(voxelSize) / advection.getMaxVelocity();
                    last = advection.advect<FloatGrid, tools::BoxSampler>(*fog, dt);
                });
            Digest digest;
            digest.addGrid(*last);
            context.record(tool, name, timing, digest);
        }
    }
}

/// The values of one float attribute of every point, in leaf order
void
addAttribute(Digest& digest, const points::PointDataGrid& grid, const std::string& attribute)
{
    for (auto leaf = grid.tree().cbeginLeaf(); leaf; ++leaf) {
        points::AttributeHandle<float> handle(leaf->constAttributeArray(attribute));
        for (Index i = 0; i < handle.size(); ++i) {
            digest.add(handle.get(i));
        }
        digest.add(handle.size());
    }
}

void
runPointSampling(Context& context)
{
    const float voxelSize = kSamplingVoxelSize;
    FloatGrid::Ptr sphere = makeSphere<FloatGrid>(voxelSize);
    points::PointDataGrid::Ptr scattered = points::denseUniformPointScatter(*sphere, 8.0f);

    // The source with the transform of the points, and one in another transform
    FloatGrid::Ptr same = makeFogSphere(voxelSize);
    FloatGrid::Ptr other = makeFogSphere(1.5f * voxelSize);
    other->transform().postTranslate(Vec3d(voxelSize / 3.0f));

    struct Source { const char* name; const FloatGrid* grid; };
    const Source sources[] = {{"same transform", same.get()}, {"source in another transform", other.get()}};
    enum Kind { Box, Quadratic, Closest };
    const std::pair<const char*, Kind> kinds[] = {
        {"points::boxSample", Box}, {"points::quadraticSample", Quadratic}, {"points::pointSample", Closest}};

    for (const auto& kind : kinds) {
        for (const Source& source : sources) {
            const std::string tool = kind.first;
            const std::string name = source.name;
            if (!context.wanted(tool, name)) continue;
            context.progress(tool, name);
            points::PointDataGrid::Ptr last;
            const Timing timing = context.time(
                [&]() { return scattered->deepCopy(); },
                [&](points::PointDataGrid::Ptr& grid) {
                    switch (kind.second) {
                        case Box: points::boxSample(*grid, *source.grid, "sampled"); break;
                        case Quadratic: points::quadraticSample(*grid, *source.grid, "sampled"); break;
                        default: points::pointSample(*grid, *source.grid, "sampled"); break;
                    }
                    last = grid;
                });
            Digest digest;
            addAttribute(digest, *last, "sampled");
            context.record(tool, name, timing, digest);
        }
    }
}

} // anonymous namespace

void
runSampling(Context& context)
{
    runResample(context);
    runMultiRes(context);
    runVolumeAdvection(context);
    runPointSampling(context);
}

} // namespace bench
