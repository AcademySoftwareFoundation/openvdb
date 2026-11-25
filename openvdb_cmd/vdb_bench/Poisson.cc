// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file Poisson.cc
///
/// @brief Group poisson: the Laplacian matrix builders and the solver.

#include "Harness.h"

#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/PoissonSolver.h>

namespace bench {

namespace {

using IndexTree = FloatTree::ValueConverter<tools::poisson::VIndex>::Type;

struct Domain
{
    IndexTree::Ptr index;
    BoolTree::Ptr interior;
};

Domain
makeDomain(const FloatGrid& grid)
{
    Domain domain;
    domain.index = tools::poisson::createIndexTree(grid.tree());
    domain.interior.reset(new BoolTree(*domain.index, /*background=*/false, TopologyCopy()));
    tools::erodeActiveValues(*domain.interior, /*iterations=*/1, tools::NN_FACE, tools::IGNORE_TILES);
    return domain;
}

void
addMatrix(Digest& digest, const tools::poisson::LaplacianMatrix& matrix)
{
    digest.add(matrix.numRows());
    for (math::pcg::SizeType row = 0; row < matrix.numRows(); ++row) {
        const auto constRow = matrix.getConstRow(row);
        digest.add(constRow.size());
        for (auto iter = constRow.cbegin(); iter; ++iter) {
            digest.add(iter.column());
            digest.add(*iter);
        }
    }
}

} // anonymous namespace

void
runPoisson(Context& context)
{
    // The cost of the matrix does not depend on the voxel size, so the domain is fixed
    const int dimension = 128;
    FloatGrid::Ptr box = makeDenseBox(dimension, 1.0f);

    // The same box with the voxels outside a sphere switched off
    FloatGrid::Ptr ball = box->deepCopy();
    const Vec3d center(0.5 * (dimension - 1));
    for (auto iter = box->cbeginValueOn(); iter; ++iter) {
        if ((iter.getCoord().asVec3d() - center).length() > 0.47 * dimension) {
            ball->tree().setValueOff(iter.getCoord());
        }
    }

    const Domain boxDomain = makeDomain(*box);
    const Domain ballDomain = makeDomain(*ball);

    const std::string tool = "poisson::createISLaplacian";
    for (int staggered = 0; staggered < 2; ++staggered) {
        const std::string name = std::string("box 128 ") + (staggered ? "staggered" : "not staggered");
        if (!context.wanted(tool, name)) continue;
        context.progress(tool, name);
        tools::poisson::LaplacianMatrix::Ptr matrix;
        const Timing timing = context.time(
            []() { return 0; },
            [&](int&) { matrix = tools::poisson::createISLaplacian(
                *boxDomain.index, *boxDomain.interior, staggered != 0); });
        Digest digest;
        addMatrix(digest, *matrix);
        context.record(tool, name, timing, digest);
    }

    {
        const std::string toolWithBoundary = "poisson::createISLaplacianWithBoundaryConditions";
        const std::string name = "ball 128 staggered";
        if (context.wanted(toolWithBoundary, name)) {
            context.progress(toolWithBoundary, name);
            const tools::poisson::DirichletBoundaryOp<tools::poisson::LaplacianMatrix::ValueType> boundary;
            const math::pcg::SizeType size = math::pcg::SizeType(ballDomain.index->activeVoxelCount());
            tools::poisson::LaplacianMatrix::Ptr matrix;
            math::pcg::Vector<tools::poisson::LaplacianMatrix::ValueType> source(size, 0.0);
            const Timing timing = context.time(
                []() { return 0; },
                [&](int&) {
                    matrix = tools::poisson::createISLaplacianWithBoundaryConditions(
                        *ballDomain.index, *ballDomain.interior, boundary, source, /*staggered=*/true);
                });
            Digest digest;
            addMatrix(digest, *matrix);
            context.record(toolWithBoundary, name, timing, digest);
        }
    }

    {
        const std::string toolSolve = "poisson::solve";
        const std::string name = "box 128 30 iterations";
        if (context.wanted(toolSolve, name)) {
            context.progress(toolSolve, name);
            FloatTree::Ptr solution;
            int iterations = 0;
            const Timing timing = context.time(
                []() { return 0; },
                [&](int&) {
                    // A fixed number of iterations, so that the work does not depend on rounding
                    math::pcg::State state = math::pcg::terminationDefaults<float>();
                    state.iterations = 30;
                    state.relativeError = 0.0;
                    state.absoluteError = 0.0;
                    solution = tools::poisson::solve(box->tree(), state);
                    iterations = state.iterations;
                });
            Digest digest;
            digest.add(iterations);
            double sum = 0.0;
            for (auto iter = solution->cbeginValueOn(); iter; ++iter) sum += double(*iter);
            digest.addApproximate(sum);
            digest.add(solution->activeVoxelCount());
            context.record(toolSolve, name, timing, digest);
        }
    }
}

} // namespace bench
