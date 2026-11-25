// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file Filter.cc
///
/// @brief Group filter: tools::Filter.

#include "Harness.h"

#include <openvdb/tools/Filter.h>

#include <functional>
#include <map>

namespace bench {

void
runFilter(Context& context)
{
    using FilterT = tools::Filter<FloatGrid>;
    struct Entry { std::string name; int width; bool masked; std::function<void(FilterT&, const FloatGrid*)> apply; };
    const std::vector<Entry> entries = {
        {"mean 1", 1, false, [](FilterT& filter, const FloatGrid*) { filter.mean(1); }},
        {"mean 3", 3, false, [](FilterT& filter, const FloatGrid*) { filter.mean(3); }},
        {"mean 9", 9, false, [](FilterT& filter, const FloatGrid*) { filter.mean(9); }},
        {"gaussian 1", 1, false, [](FilterT& filter, const FloatGrid*) { filter.gaussian(1); }},
        {"gaussian 3", 3, false, [](FilterT& filter, const FloatGrid*) { filter.gaussian(3); }},
        {"gaussian 9", 9, false, [](FilterT& filter, const FloatGrid*) { filter.gaussian(9); }},
        {"median 1", 1, false, [](FilterT& filter, const FloatGrid*) { filter.median(1); }},
        {"median 9", 9, false, [](FilterT& filter, const FloatGrid*) { filter.median(9); }},
        {"median 12", 12, false, [](FilterT& filter, const FloatGrid*) { filter.median(12); }},
        {"mean 1 with mask", 1, true,
            [](FilterT& filter, const FloatGrid* mask) { filter.mean(1, 1, mask); }}};

    const std::string tool = "tools::Filter";
    const float fine = context.options().voxelSize;

    // The median reads thousands of voxels each, so the wide ones run on a coarser grid
    std::map<int, FloatGrid::Ptr> spheres;
    for (const Entry& entry : entries) {
        const float voxelSize = (entry.width >= 9 && entry.name.rfind("median", 0) == 0) ? 4.0f * fine : fine;
        const std::string name = entry.name + " " + voxelLabel(voxelSize);
        if (!context.wanted(tool, name)) continue;
        context.progress(tool, name);
        FloatGrid::Ptr& fog = spheres[int(std::lround(1.0f / voxelSize))];
        if (!fog) fog = makeFogSphere(voxelSize);
        FloatGrid::Ptr last;
        const Timing timing = context.time(
            [&]() { return fog->deepCopy(); },
            [&](FloatGrid::Ptr& grid) {
                FilterT filter(*grid);
                entry.apply(filter, entry.masked ? fog.get() : nullptr);
                last = grid;
            });
        Digest digest;
        digest.addGrid(*last);
        context.record(tool, name, timing, digest);
    }

    // The same filter on a bool grid, which has its own accumulate path
    {
        const std::string name = "mean 1 BoolGrid " + voxelLabel(fine);
        if (context.wanted(tool, name)) {
            context.progress(tool, name);
            FloatGrid::Ptr sphere = makeSphere<FloatGrid>(fine);
            BoolGrid::Ptr source = BoolGrid::create(false);
            source->setTransform(sphere->transform().copy());
            for (auto iter = sphere->cbeginValueOn(); iter; ++iter) {
                source->tree().setValueOn(iter.getCoord(), *iter < 0.0f);
            }
            BoolGrid::Ptr last;
            const Timing timing = context.time(
                [&]() { return source->deepCopy(); },
                [&](BoolGrid::Ptr& grid) {
                    tools::Filter<BoolGrid> filter(*grid);
                    filter.mean(1);
                    last = grid;
                });
            Digest digest;
            digest.addGrid(*last);
            context.record(tool, name, timing, digest);
        }
    }
}

} // namespace bench
