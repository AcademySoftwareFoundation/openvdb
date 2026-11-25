// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file Operators.cc
///
/// @brief Group operators: the grid operators of tools/GridOperators.h.

#include "Harness.h"

#include <openvdb/tools/GridOperators.h>

namespace bench {

namespace {

struct TransformCase
{
    std::string name;
    math::Transform::Ptr transform;
};

std::vector<TransformCase>
makeTransforms(const FloatGrid& sphere)
{
    const double voxelSize = sphere.voxelSize()[0];
    std::vector<TransformCase> cases;
    cases.push_back({"uniform", sphere.transform().copy()});

    // Each of these goes through a different case of the map dispatch
    math::Transform::Ptr scale = math::Transform::createLinearTransform(voxelSize);
    scale->preScale(Vec3d(1.0, 1.5, 2.0));
    cases.push_back({"scale 1 1.5 2", scale});

    const CoordBBox bounds = sphere.evalActiveVoxelBoundingBox();
    const BBoxd box(bounds.min().asVec3d(), bounds.max().asVec3d());
    cases.push_back({"frustum", math::Transform::createFrustumTransform(
        box, 0.5, bounds.dim()[2] * voxelSize, voxelSize)});
    return cases;
}

} // anonymous namespace

void
runOperators(Context& context)
{
    const float voxelSize = context.options().voxelSize;
    FloatGrid::Ptr sphere = makeSphere<FloatGrid>(voxelSize);
    Vec3SGrid::Ptr field = makeRadialField(*sphere);

    // Half of the sphere
    CoordBBox half = sphere->evalActiveVoxelBoundingBox();
    half.max()[0] = (half.min()[0] + half.max()[0]) / 2;
    BoolGrid::Ptr mask = BoolGrid::create(false);
    mask->setTransform(sphere->transform().copy());
    mask->tree().fill(half, true, true);

    for (const TransformCase& transformCase : makeTransforms(*sphere)) {
        FloatGrid::Ptr scalar = sphere->deepCopy();
        scalar->setTransform(transformCase.transform->copy());
        Vec3SGrid::Ptr vector = field->deepCopy();
        vector->setTransform(transformCase.transform->copy());

        auto row = [&](const std::string& tool, auto&& compute) {
            const std::string name = transformCase.name + " " + voxelLabel(voxelSize);
            if (!context.wanted(tool, name)) return;
            context.progress(tool, name);
            Digest digest;
            // The result type comes from compute
            auto result = compute();
            const Timing timing = context.time(
                []() { return 0; }, [&](int&) { result = compute(); });
            digest.addGrid(*result);
            context.record(tool, name, timing, digest);
        };

        row("gradient", [&]() { return tools::gradient(*scalar); });
        row("laplacian", [&]() { return tools::laplacian(*scalar); });
        row("meanCurvature", [&]() { return tools::meanCurvature(*scalar); });
        row("cpt", [&]() { return tools::cpt(*scalar); });
        row("divergence", [&]() { return tools::divergence(*vector); });
        row("curl", [&]() { return tools::curl(*vector); });
        row("magnitude", [&]() { return tools::magnitude(*vector); });
        row("normalize", [&]() { return tools::normalize(*vector); });
        if (transformCase.name == "uniform") {
            row("gradient with mask", [&]() { return tools::gradient(*scalar, *mask); });
        }
    }
}

} // namespace bench
