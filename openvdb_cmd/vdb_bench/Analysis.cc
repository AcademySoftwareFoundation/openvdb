// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file Analysis.cc
///
/// @brief Group analysis: LevelSetMeasure, FastSweeping, Diagnose and the statistics tools.

#include "Harness.h"

#include <openvdb/math/Operators.h>
#include <openvdb/tools/Diagnostics.h>
#include <openvdb/tools/FastSweeping.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <openvdb/tools/Statistics.h>

namespace bench {

namespace {

void
runMeasure(Context& context)
{
    const float voxelSize = context.options().voxelSize;
    std::vector<NamedGrid> grids;
    grids.push_back({"sphere", makeSphere<FloatGrid>(voxelSize)});
    grids.push_back({"icosahedron", makeIcosahedron(voxelSize)});

    enum Quantity { Area, Volume, MeanCurvature };
    const std::pair<const char*, Quantity> quantities[] = {
        {"area", Area}, {"volume", Volume}, {"totMeanCurvature", MeanCurvature}};

    const std::string tool = "LevelSetMeasure";
    for (const NamedGrid& named : grids) {
        for (const auto& quantity : quantities) {
            const std::string name = std::string(quantity.first) + " " + named.name;
            if (!context.wanted(tool, name)) continue;
            context.progress(tool, name);
            double result = 0.0;
            const Timing timing = context.time(
                []() { return 0; },
                [&](int&) {
                    tools::LevelSetMeasure<FloatGrid> measurer(*named.grid);
                    switch (quantity.second) {
                        case Area: result = measurer.area(); break;
                        case Volume: result = measurer.volume(); break;
                        default: result = measurer.totMeanCurvature(); break;
                    }
                });
            Digest digest;
            digest.addApproximate(result);
            context.record(tool, name, timing, digest);
        }
    }
}

void
runFastSweeping(Context& context)
{
    const float voxelSize = context.options().voxelSize;
    FloatGrid::Ptr sphere = distort(makeSphere<FloatGrid>(voxelSize), voxelSize);
    {
        const std::string tool = "FastSweeping::sdfToSdf";
        const std::string name = "distorted sphere";
        if (context.wanted(tool, name)) {
            context.progress(tool, name);
            FloatGrid::Ptr result;
            const Timing timing = context.time(
                []() { return 0; },
                [&](int&) { result = tools::sdfToSdf(*sphere); });
            Digest digest;
            digest.addGrid(*result);
            context.record(tool, name, timing, digest);
        }
    }
    {
        const std::string tool = "FastSweeping::sdfToExt";
        const std::string name = "distorted sphere scalar extension";
        if (context.wanted(tool, name)) {
            context.progress(tool, name);
            FloatGrid::Ptr result;
            const auto extension = [](const Vec3R& xyz) { return float(xyz[0] + 2.0 * xyz[1] - xyz[2]); };
            const Timing timing = context.time(
                []() { return 0; },
                [&](int&) { result = tools::sdfToExt(*sphere, extension, 0.0f); });
            Digest digest;
            digest.addGrid(*result);
            context.record(tool, name, timing, digest);
        }
    }
}

void
runDiagnose(Context& context)
{
    const float voxelSize = context.options().voxelSize;
    FloatGrid::Ptr sphere = makeSphere<FloatGrid>(voxelSize);
    const std::string tool = "Diagnose::check";

    for (int check = 0; check < 2; ++check) {
        const std::string name = std::string(check == 0 ? "CheckEikonal" : "CheckNormGrad") + " sphere";
        if (!context.wanted(tool, name)) continue;
        context.progress(tool, name);
        std::string message;
        Index64 failures = 0;
        const Timing timing = context.time(
            []() { return 0; },
            [&](int&) {
                tools::Diagnose<FloatGrid> diagnose(*sphere);
                if (check == 0) {
                    message = diagnose.check(tools::CheckEikonal<FloatGrid>(*sphere, 0.5f, 1.5f),
                        false, true, false, false);
                } else {
                    message = diagnose.check(tools::CheckNormGrad<FloatGrid>(*sphere, 0.5f, 1.5f),
                        false, true, false, false);
                }
                failures = diagnose.failureCount();
            });
        Digest digest;
        for (char c : message) digest.add(c);
        digest.add(failures);
        context.record(tool, name, timing, digest);
    }
}

void
addStats(Digest& digest, const math::Stats& stats)
{
    digest.add(stats.size());
    digest.addApproximate(stats.min());
    digest.addApproximate(stats.max());
    digest.addApproximate(stats.mean());
    digest.addApproximate(stats.variance());
}

void
runStatistics(Context& context)
{
    const float voxelSize = context.options().voxelSize;
    FloatGrid::Ptr sphere = makeSphere<FloatGrid>(voxelSize);
    Vec3SGrid::Ptr field = makeRadialField(*sphere);

    // A grid with active tiles. The fix for active tiles in the statistics tools changes
    // the result, so these rows are expected to differ between the builds.
    FloatGrid::Ptr tiles = FloatGrid::create(0.0f);
    tiles->setTransform(math::Transform::createLinearTransform(voxelSize));
    tiles->tree().fill(CoordBBox(Coord(0), Coord(255)), 1.0f, true);
    tiles->tree().setValueOn(Coord(300, 5, 5), 2.0f);
    tiles->tree().setValueOn(Coord(5, 300, 5), 3.0f);

    using MapType = math::UniformScaleMap;
    using GradientOp = math::Gradient<MapType, math::FD_1ST>;
    using MagnitudeOp = math::OpMagnitude<GradientOp, MapType>;
    using GradientCompound = math::MapAdapter<MapType, MagnitudeOp, double>;
    using DivergenceOp = math::Divergence<MapType, math::FD_1ST>;
    using DivergenceCompound = math::MapAdapter<MapType, DivergenceOp, double>;

    const MapType::ConstPtr floatMap = sphere->constTransform().constMap<MapType>();
    const MapType::ConstPtr vectorMap = field->constTransform().constMap<MapType>();
    if (!floatMap || !vectorMap) throw std::runtime_error("the statistics rows need a uniform scale map");

    {
        const std::string tool = "opStatistics";
        auto row = [&](const std::string& name, auto&& statistics) {
            if (!context.wanted(tool, name)) return;
            context.progress(tool, name);
            math::Stats stats;
            const Timing timing = context.time(
                []() { return 0; }, [&](int&) { stats = statistics(); });
            Digest digest;
            addStats(digest, stats);
            context.record(tool, name, timing, digest);
        };
        row("gradient magnitude MapAdapter", [&]() {
            return tools::opStatistics(sphere->cbeginValueOn(), GradientCompound(*floatMap)); });
        row("divergence MapAdapter", [&]() {
            return tools::opStatistics(field->cbeginValueOn(), DivergenceCompound(*vectorMap)); });
        row("divergence index space", [&]() {
            return tools::opStatistics(field->cbeginValueOn(), math::ISDivergence<math::FD_1ST>()); });
        row("laplacian index space active tiles", [&]() {
            return tools::opStatistics(tiles->cbeginValueOn(), math::ISLaplacian<math::CD_SECOND>()); });
    }
    {
        const std::string tool = "opExtrema";
        auto row = [&](const std::string& name, auto&& extrema) {
            if (!context.wanted(tool, name)) return;
            context.progress(tool, name);
            math::Extrema result;
            const Timing timing = context.time(
                []() { return 0; }, [&](int&) { result = extrema(); });
            Digest digest;
            digest.add(result.size());
            digest.addApproximate(result.min());
            digest.addApproximate(result.max());
            context.record(tool, name, timing, digest);
        };
        row("gradient magnitude MapAdapter", [&]() {
            return tools::opExtrema(sphere->cbeginValueOn(), GradientCompound(*floatMap)); });
        row("laplacian index space", [&]() {
            return tools::opExtrema(sphere->cbeginValueOn(), math::ISLaplacian<math::CD_SECOND>()); });
        row("laplacian index space active tiles", [&]() {
            return tools::opExtrema(tiles->cbeginValueOn(), math::ISLaplacian<math::CD_SECOND>()); });
    }
}

} // anonymous namespace

void
runAnalysis(Context& context)
{
    runMeasure(context);
    runFastSweeping(context);
    runDiagnose(context);
    runStatistics(context);
}

} // namespace bench
