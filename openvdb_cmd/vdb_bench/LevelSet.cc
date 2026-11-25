// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file LevelSet.cc
///
/// @brief Group levelset: the tracker, advection, morphing and the level set filters.

#include "Harness.h"

#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetMorph.h>
#include <openvdb/tools/LevelSetTracker.h>

#if defined(__has_include)
#if __has_include(<openvdb/tree/NeighborAccessor.h>)
#include <openvdb/tree/NeighborAccessor.h>
#define BENCH_HAS_NEIGHBOR_ACCESSOR 1
#endif
#endif

#include <functional>
#include <map>

namespace bench {

namespace {

const math::BiasedGradientScheme kSchemes[] = {
    math::FIRST_BIAS, math::SECOND_BIAS, math::THIRD_BIAS, math::HJWENO5_BIAS};

const char*
schemeName(math::BiasedGradientScheme scheme)
{
    switch (scheme) {
        case math::FIRST_BIAS: return "FIRST";
        case math::SECOND_BIAS: return "SECOND";
        case math::THIRD_BIAS: return "THIRD";
        default: return "HJWENO5";
    }
}

void
runNormalize(Context& context)
{
    const float voxelSize = context.options().voxelSize;
    std::vector<NamedGrid> grids;
    grids.push_back({"distorted sphere", distort(makeSphere<FloatGrid>(voxelSize), voxelSize)});
    grids.push_back({"distorted icosahedron", distort(makeIcosahedron(voxelSize), voxelSize)});
    grids.push_back({"distorted cube", distort(makeCube(voxelSize), voxelSize)});

    const std::string tool = "LevelSetTracker::normalize";
    for (const NamedGrid& named : grids) {
        for (math::TemporalIntegrationScheme temporal : {math::TVD_RK1, math::TVD_RK3}) {
            for (math::BiasedGradientScheme scheme : kSchemes) {
                const std::string name = named.name + " " + schemeName(scheme) + " RK" +
                    std::to_string(int(temporal) + 1) + " " + voxelLabel(voxelSize);
                if (!context.wanted(tool, name)) continue;
                context.progress(tool, name);
                FloatGrid::Ptr last;
                const Timing timing = context.time(
                    [&]() { return named.grid->deepCopy(); },
                    [&](FloatGrid::Ptr& grid) {
                        tools::LevelSetTracker<FloatGrid> tracker(*grid);
                        tracker.setSpatialScheme(scheme);
                        tracker.setTemporalScheme(temporal);
                        tracker.setNormCount(10);
                        tracker.normalize();
                        last = grid;
                    });
                Digest digest;
                digest.addGrid(*last);
                context.record(tool, name, timing, digest);
            }
        }
    }
}

/// Advect 0 to 0.25 in 10 slices. A slice at a time lets a run that blows up stop early.
template<typename FieldT>
void
advectRow(Context& context, const std::string& tool, const std::string& name,
    const FloatGrid& source, const FieldT& field, math::BiasedGradientScheme scheme)
{
    if (!context.wanted(tool, name)) return;
    context.progress(tool, name);
    FloatGrid::Ptr last;
    size_t steps = 0;
    bool unstable = false;
    const Timing timing = context.time(
        [&]() { return source.deepCopy(); },
        [&](FloatGrid::Ptr& grid) {
            tools::LevelSetAdvection<FloatGrid, FieldT> advection(*grid, field);
            advection.setSpatialScheme(scheme);
            advection.setTemporalScheme(math::TVD_RK2);
            advection.setTrackerSpatialScheme(scheme);
            advection.setTrackerTemporalScheme(math::TVD_RK2);
            const Index64 limit = 4 * grid->activeVoxelCount();
            const int slices = 10;
            steps = 0;
            unstable = false;
            for (int slice = 0; slice < slices && !unstable; ++slice) {
                steps += advection.advect(0.25f * float(slice) / float(slices),
                    0.25f * float(slice + 1) / float(slices));
                unstable = grid->activeVoxelCount() > limit;
            }
            last = grid;
        });
    Digest digest;
    digest.addGrid(*last);
    digest.add(steps);
    digest.add(unstable);
    context.record(tool, name, timing, digest);
}

void
runAdvection(Context& context)
{
    // The voxel size is smaller than for the other rows to keep the run time down
    const float voxelSizes[] = {1.0f / 256.0f, 1.0f / 512.0f};
    for (float voxelSize : voxelSizes) {
        FloatGrid::Ptr sphere = makeSphere<FloatGrid>(voxelSize);
        const tools::EnrightField<float> field;
        for (math::BiasedGradientScheme scheme : kSchemes) {
            advectRow(context, "LevelSetAdvection", std::string("enright ") + schemeName(scheme) +
                " " + voxelLabel(voxelSize), *sphere, field, scheme);
        }
    }

    // Isolates the stencil change from the field: a velocity grid, sampled by the default
    // field and, where the library has it, by a NeighborAccessor.
    const float voxelSize = 1.0f / 256.0f;
    FloatGrid::Ptr sphere = makeSphere<FloatGrid>(voxelSize);
    Vec3SGrid::Ptr velocity = makeVelocityGrid();
    const std::string name = "velocity grid HJWENO5 " + voxelLabel(voxelSize);
    {
        const tools::DiscreteField<Vec3SGrid> field(*velocity);
        advectRow(context, "LevelSetAdvection DiscreteField", name, *sphere, field, math::HJWENO5_BIAS);
    }
#ifdef BENCH_HAS_NEIGHBOR_ACCESSOR
    {
        using Field = tools::DiscreteField<Vec3SGrid, tools::BoxSampler,
            tree::NeighborAccessor<const Vec3SGrid::TreeType>>;
        const Field field(*velocity);
        advectRow(context, "DiscreteField NeighborAccessor", name, *sphere, field, math::HJWENO5_BIAS);
    }
#endif
}

void
runMorph(Context& context)
{
    const float voxelSize = context.options().voxelSize;
    FloatGrid::Ptr source = makeSphere<FloatGrid>(voxelSize);
    FloatGrid::Ptr target = tools::createLevelSetIcosahedron<FloatGrid>(
        0.2f, Vec3f(0.35f), voxelSize, 3.0f);
    FloatGrid::Ptr targetOther = tools::createLevelSetIcosahedron<FloatGrid>(
        0.2f, Vec3f(0.35f), 1.5f * voxelSize, 3.0f);
    targetOther->transform().postTranslate(Vec3d(voxelSize / 3.0f));
    FloatGrid::Ptr mask = makeFogSphere(voxelSize);

    struct Variant { const char* name; const FloatGrid* target; const FloatGrid* mask; };
    const Variant variants[] = {
        {"same transform", target.get(), nullptr},
        {"target in another transform", targetOther.get(), nullptr},
        {"alpha mask", target.get(), mask.get()}};

    const std::string tool = "LevelSetMorphing";
    for (const Variant& variant : variants) {
        const std::string name = std::string(variant.name) + " " + voxelLabel(voxelSize);
        if (!context.wanted(tool, name)) continue;
        context.progress(tool, name);
        FloatGrid::Ptr last;
        size_t steps = 0;
        const Timing timing = context.time(
            [&]() { return source->deepCopy(); },
            [&](FloatGrid::Ptr& grid) {
                tools::LevelSetMorphing<FloatGrid> morph(*grid, *variant.target);
                if (variant.mask) morph.setAlphaMask(*variant.mask);
                morph.setSpatialScheme(math::HJWENO5_BIAS);
                morph.setTemporalScheme(math::TVD_RK2);
                morph.setTrackerSpatialScheme(math::HJWENO5_BIAS);
                morph.setTrackerTemporalScheme(math::TVD_RK2);
                steps = morph.advect(0.0f, 0.02f);
                last = grid;
            });
        Digest digest;
        digest.addGrid(*last);
        digest.add(steps);
        context.record(tool, name, timing, digest);
    }
}

void
runLevelSetFilter(Context& context)
{
    using FilterT = tools::LevelSetFilter<FloatGrid>;
    struct Entry { std::string name; int width; std::function<void(FilterT&)> apply; };
    const std::vector<Entry> entries = {
        {"meanCurvature", 0, [](FilterT& filter) { filter.meanCurvature(); }},
        {"laplacian", 0, [](FilterT& filter) { filter.laplacian(); }},
        {"fillet", 0, [](FilterT& filter) { filter.fillet(); }},
        {"gaussian 1", 1, [](FilterT& filter) { filter.gaussian(1); }},
        {"mean 1", 1, [](FilterT& filter) { filter.mean(1); }},
        {"mean 3", 3, [](FilterT& filter) { filter.mean(3); }},
        {"mean 9", 9, [](FilterT& filter) { filter.mean(9); }},
        {"median 1", 1, [](FilterT& filter) { filter.median(1); }},
        {"median 9", 9, [](FilterT& filter) { filter.median(9); }},
        {"median 12", 12, [](FilterT& filter) { filter.median(12); }},
        {"offset", 0, [](FilterT& filter) { filter.offset(0.001f); }}};

    // The wide filters read hundreds to thousands of voxels each, so they run on a grid
    // that is four times coarser.
    std::map<int, FloatGrid::Ptr> spheres;
    const std::string tool = "LevelSetFilter";
    for (const Entry& entry : entries) {
        const float voxelSize = context.options().voxelSize * (entry.width >= 9 ? 4.0f : 1.0f);
        const std::string name = entry.name + " " + voxelLabel(voxelSize);
        if (!context.wanted(tool, name)) continue;
        context.progress(tool, name);
        FloatGrid::Ptr& sphere = spheres[int(std::lround(1.0f / voxelSize))];
        if (!sphere) sphere = makeSphere<FloatGrid>(voxelSize);
        FloatGrid::Ptr last;
        const Timing timing = context.time(
            [&]() { return sphere->deepCopy(); },
            [&](FloatGrid::Ptr& grid) {
                FilterT filter(*grid);
                filter.setSpatialScheme(math::HJWENO5_BIAS);
                filter.setTemporalScheme(math::TVD_RK1);
                filter.setNormCount(1);
                entry.apply(filter);
                last = grid;
            });
        Digest digest;
        digest.addGrid(*last);
        context.record(tool, name, timing, digest);
    }
}

} // anonymous namespace

void
runLevelSet(Context& context)
{
    runNormalize(context);
    runAdvection(context);
    runMorph(context);
    runLevelSetFilter(context);
}

} // namespace bench
