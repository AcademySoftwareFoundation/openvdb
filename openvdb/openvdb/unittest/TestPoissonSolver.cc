// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file unittest/TestPoissonSolver.cc
/// @authors D.J. Hill, Peter Cucka

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/math/ConjGradient.h> // for JacobiPreconditioner
#include <openvdb/tools/Composite.h> // for csgDifference/Union/Intersection
#include <openvdb/tools/LevelSetSphere.h> // for tools::createLevelSetSphere()
#include <openvdb/tools/LevelSetUtil.h> // for tools::sdfToFogVolume()
#include <openvdb/tools/MeshToVolume.h> // for createLevelSetBox()
#include <openvdb/tools/PoissonSolver.h>
#include <openvdb/tools/GridOperators.h> // for divergence and gradient

#include <gtest/gtest.h>

#include <cmath>


class TestPoissonSolver: public ::testing::Test
{
};


////////////////////////////////////////


TEST_F(TestPoissonSolver, testIndexTree)
{
    using namespace openvdb;
    using tools::poisson::VIndex;

    using VIdxTree = FloatTree::ValueConverter<VIndex>::Type;
    using LeafNodeType = VIdxTree::LeafNodeType;

    VIdxTree tree;
    /// @todo populate tree
    tree::LeafManager<const VIdxTree> leafManager(tree);

    VIndex testOffset = 0;
    for (size_t n = 0, N = leafManager.leafCount(); n < N; ++n) {
        const LeafNodeType& leaf = leafManager.leaf(n);
        for (LeafNodeType::ValueOnCIter it = leaf.cbeginValueOn(); it; ++it, testOffset++) {
            EXPECT_EQ(testOffset, *it);
        }
    }

    //if (testOffset != VIndex(tree.activeVoxelCount())) {
    //    std::cout << "--Testing offsetmap - "
    //              << testOffset<<" != "
    //              << tree.activeVoxelCount()
    //              << " has active tile count "
    //              << tree.activeTileCount()<<std::endl;
    //}

    EXPECT_EQ(VIndex(tree.activeVoxelCount()), testOffset);
}


TEST_F(TestPoissonSolver, testTreeToVectorToTree)
{
    using namespace openvdb;
    using tools::poisson::VIndex;

    using VIdxTree = FloatTree::ValueConverter<VIndex>::Type;

    FloatGrid::Ptr sphere = tools::createLevelSetSphere<FloatGrid>(
        /*radius=*/10.f, /*center=*/Vec3f(0.f), /*voxelSize=*/0.25f);
    tools::sdfToFogVolume(*sphere);
    FloatTree& inputTree = sphere->tree();

    const Index64 numVoxels = inputTree.activeVoxelCount();

    // Generate an index tree.
    VIdxTree::Ptr indexTree = tools::poisson::createIndexTree(inputTree);
    EXPECT_TRUE(bool(indexTree));

    // Copy the values of the active voxels of the tree into a vector.
    math::pcg::VectorS::Ptr vec =
        tools::poisson::createVectorFromTree<float>(inputTree, *indexTree);
    EXPECT_EQ(math::pcg::SizeType(numVoxels), vec->size());

    {
        // Convert the vector back to a tree.
        FloatTree::Ptr inputTreeCopy = tools::poisson::createTreeFromVector(
            *vec, *indexTree, /*bg=*/0.f);

        // Check that voxel values were preserved.
        FloatGrid::ConstAccessor inputAcc = sphere->getConstAccessor();
        for (FloatTree::ValueOnCIter it = inputTreeCopy->cbeginValueOn(); it; ++it) {
            const Coord ijk = it.getCoord();
            //if (!math::isApproxEqual(*it, inputTree.getValue(ijk))) {
            //    std::cout << " value error " << *it << " "
            //        << inputTree.getValue(ijk) << std::endl;
            //}
            EXPECT_NEAR(inputAcc.getValue(ijk), *it, /*tolerance=*/1.0e-6);
        }
    }
}


TEST_F(TestPoissonSolver, testLaplacian)
{
    using namespace openvdb;
    using tools::poisson::VIndex;

    using VIdxTree = FloatTree::ValueConverter<VIndex>::Type;

    // For two different problem sizes, N = 8 and N = 20...
    for (int N = 8; N <= 20; N += 12) {
        // Construct an N x N x N volume in which the value of voxel (i, j, k)
        // is sin(i) * sin(j) * sin(k), using a voxel spacing of pi / N.
        const double delta = openvdb::math::pi<double>() / N;
        FloatTree inputTree(/*background=*/0.f);
        Coord ijk(0);
        Int32 &i = ijk[0], &j = ijk[1], &k = ijk[2];
        for (i = 1; i < N; ++i) {
            for (j = 1; j < N; ++j) {
                for (k = 1; k < N; ++k) {
                    inputTree.setValue(ijk, static_cast<float>(
                        std::sin(delta * i) * std::sin(delta * j) * std::sin(delta * k)));
                }
            }
        }
        const Index64 numVoxels = inputTree.activeVoxelCount();

        // Generate an index tree.
        VIdxTree::Ptr indexTree = tools::poisson::createIndexTree(inputTree);
        EXPECT_TRUE(bool(indexTree));

        // Copy the values of the active voxels of the tree into a vector.
        math::pcg::VectorS::Ptr source =
            tools::poisson::createVectorFromTree<float>(inputTree, *indexTree);
        EXPECT_EQ(math::pcg::SizeType(numVoxels), source->size());

        // Create a mask of the interior voxels of the source tree.
        BoolTree interiorMask(/*background=*/false);
        interiorMask.fill(CoordBBox(Coord(2), Coord(N-2)), /*value=*/true, /*active=*/true);

        // Compute the Laplacian of the source:
        //     D^2 sin(i) * sin(j) * sin(k) = -3 sin(i) * sin(j) * sin(k)
        tools::poisson::LaplacianMatrix::Ptr laplacian =
            tools::poisson::createISLaplacian(*indexTree, interiorMask, /*staggered=*/true);
        laplacian->scale(1.0 / (delta * delta)); // account for voxel spacing
        EXPECT_EQ(math::pcg::SizeType(numVoxels), laplacian->size());

        math::pcg::VectorS result(source->size());
        laplacian->vectorMultiply(*source, result);

        // Dividing the result by the source should produce a vector of uniform value -3.
        // Due to finite differencing, the actual ratio will be somewhat different, though.
        const math::pcg::VectorS& src = *source;
        const float expected = // compute the expected ratio using one of the corner voxels
            float((3.0 * src[1] - 6.0 * src[0]) / (delta * delta * src[0]));
        for (math::pcg::SizeType n = 0; n < result.size(); ++n) {
            result[n] /= src[n];
            EXPECT_NEAR(expected, result[n], /*tolerance=*/1.0e-4);
        }
    }
}


TEST_F(TestPoissonSolver, testSolve)
{
    using namespace openvdb;

    FloatGrid::Ptr sphere = tools::createLevelSetSphere<FloatGrid>(
        /*radius=*/10.f, /*center=*/Vec3f(0.f), /*voxelSize=*/0.25f);
    tools::sdfToFogVolume(*sphere);

    math::pcg::State result = math::pcg::terminationDefaults<float>();
    result.iterations = 100;
    result.relativeError = result.absoluteError = 1.0e-4;

    FloatTree::Ptr outTree = tools::poisson::solve(sphere->tree(), result);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.iterations < 60);
}


////////////////////////////////////////


namespace {

struct BoundaryOp {
    void operator()(const openvdb::Coord& ijk, const openvdb::Coord& neighbor,
        double& source, double& diagonal) const
    {
        if (neighbor.x() == ijk.x() && neighbor.z() == ijk.z()) {
            // Workaround for spurious GCC 4.8 -Wstrict-overflow warning:
            const openvdb::Coord::ValueType dy = (ijk.y() - neighbor.y());
            if (dy > 0) source -= 1.0;
            else diagonal -= 1.0;
        }
    }
};


template<typename TreeType>
void
doTestSolveWithBoundaryConditions()
{
    using namespace openvdb;

    using ValueType = typename TreeType::ValueType;

    // Solve for the pressure in a cubic tank of liquid that is open at the top.
    // Boundary conditions are P = 0 at the top, dP/dy = -1 at the bottom
    // and dP/dx = 0 at the sides.
    //
    //               P = 0
    //              +------+ (N,-1,N)
    //             /|     /|
    //   (0,-1,0) +------+ |
    //            | |    | | dP/dx = 0
    //  dP/dx = 0 | +----|-+
    //            |/     |/
    // (0,-N-1,0) +------+ (N,-N-1,0)
    //           dP/dy = -1

    const int N = 9;
    const ValueType zero = zeroVal<ValueType>();
    const double epsilon = math::Delta<ValueType>::value();

    TreeType source(/*background=*/zero);
    source.fill(CoordBBox(Coord(0, -N-1, 0), Coord(N, -1, N)), /*value=*/zero);

    math::pcg::State state = math::pcg::terminationDefaults<ValueType>();
    state.iterations = 100;
    state.relativeError = state.absoluteError = epsilon;

    util::NullInterrupter interrupter;

    typename TreeType::Ptr solution = tools::poisson::solveWithBoundaryConditions(
        source, BoundaryOp(), state, interrupter, /*staggered=*/true);

    EXPECT_TRUE(state.success);
    EXPECT_TRUE(state.iterations < 60);

    // Verify that P = -y throughout the solution space.
    for (typename TreeType::ValueOnCIter it = solution->cbeginValueOn(); it; ++it) {
        EXPECT_NEAR(
            double(-it.getCoord().y()), double(*it), /*tolerance=*/10.0 * epsilon);
    }
}

} // unnamed namespace


TEST_F(TestPoissonSolver, testSolveWithBoundaryConditions)
{
    doTestSolveWithBoundaryConditions<openvdb::FloatTree>();
    doTestSolveWithBoundaryConditions<openvdb::DoubleTree>();
}


namespace {

openvdb::FloatGrid::Ptr
newCubeLS(
    const int outerLength, // in voxels
    const int innerLength, // in voxels
    const openvdb::Vec3I& centerIS, // in index space
    const float dx, // grid spacing
    bool openTop)
{
    using namespace openvdb;

    using BBox = math::BBox<Vec3f>;

    // World space dimensions and center for this box
    const float outerWS = dx * float(outerLength);
    const float innerWS = dx * float(innerLength);
    Vec3f centerWS(centerIS);
    centerWS *= dx;

    // Construct world space bounding boxes
    BBox outerBBox(
        Vec3f(-outerWS / 2, -outerWS / 2, -outerWS / 2),
        Vec3f( outerWS / 2,  outerWS / 2,  outerWS / 2));
    BBox innerBBox;
    if (openTop) {
        innerBBox = BBox(
            Vec3f(-innerWS / 2, -innerWS / 2, -innerWS / 2),
            Vec3f( innerWS / 2,  innerWS / 2,  outerWS));
    } else {
        innerBBox = BBox(
            Vec3f(-innerWS / 2, -innerWS / 2, -innerWS / 2),
            Vec3f( innerWS / 2,  innerWS / 2,  innerWS / 2));
    }
    outerBBox.translate(centerWS);
    innerBBox.translate(centerWS);

    math::Transform::Ptr xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr cubeLS = tools::createLevelSetBox<FloatGrid>(outerBBox, *xform);
    FloatGrid::Ptr inside = tools::createLevelSetBox<FloatGrid>(innerBBox, *xform);
    tools::csgDifference(*cubeLS, *inside);

    return cubeLS;
}


class LSBoundaryOp
{
public:
    LSBoundaryOp(const openvdb::FloatTree& lsTree): mLS(&lsTree) {}
    LSBoundaryOp(const LSBoundaryOp& other): mLS(other.mLS) {}

    void operator()(const openvdb::Coord& ijk, const openvdb::Coord& neighbor,
        double& source, double& diagonal) const
    {
        // Doing nothing is equivalent to imposing dP/dn = 0 boundary condition

        if (neighbor.x() == ijk.x() && neighbor.y() == ijk.y()) { // on top or bottom
            if (mLS->getValue(neighbor) <= 0.f) {
                // closed boundary
                source -= 1.0;
            } else {
                // open boundary
                diagonal -= 1.0;
            }
        }
    }

private:
    const openvdb::FloatTree* mLS;
};

} // unnamed namespace


TEST_F(TestPoissonSolver, testSolveWithSegmentedDomain)
{
    // In fluid simulations, incompressibility is enforced by the pressure, which is
    // computed as a solution of a Poisson equation.  Often, procedural animation
    // of objects (e.g., characters) interacting with liquid will result in boundary
    // conditions that describe multiple disjoint regions: regions of free surface flow
    // and regions of trapped fluid.  It is this second type of region for which
    // there may be no consistent pressure (e.g., a shrinking watertight region
    // filled with incompressible liquid).
    //
    // This unit test demonstrates how to use a level set and topological tools
    // to separate the well-posed problem of a liquid with a free surface
    // from the possibly ill-posed problem of fully enclosed liquid regions.
    //
    // For simplicity's sake, the physical boundaries are idealized as three
    // non-overlapping cubes, one with an open top and two that are fully closed.
    // All three contain incompressible liquid (x), and one of the closed cubes
    // will be partially filled so that two of the liquid regions have a free surface
    // (Dirichlet boundary condition on one side) while the totally filled cube
    // would have no free surface (Neumann boundary conditions on all sides).
    //                              ________________        ________________
    //      __            __       |   __________   |      |   __________   |
    //     |  |x x x x x |  |      |  |          |  |      |  |x x x x x |  |
    //     |  |x x x x x |  |      |  |x x x x x |  |      |  |x x x x x |  |
    //     |  |x x x x x |  |      |  |x x x x x |  |      |  |x x x x x |  |
    //     |   ——————————   |      |   ——————————   |      |   ——————————   |
    //     |________________|      |________________|      |________________|
    //
    // The first two regions are clearly well-posed, while the third region
    // may have no solution (or multiple solutions).
    // -D.J.Hill

    using namespace openvdb;

    using PreconditionerType =
        math::pcg::IncompleteCholeskyPreconditioner<tools::poisson::LaplacianMatrix>;

    // Grid spacing
    const float dx = 0.05f;

    // Construct the solid boundaries in a single grid.
    FloatGrid::Ptr solidBoundary;
    {
        // Create three non-overlapping cubes.
        const int outerDim = 41;
        const int innerDim = 31;
        FloatGrid::Ptr
            openDomain = newCubeLS(outerDim, innerDim, /*ctr=*/Vec3I(0, 0, 0), dx, /*open=*/true),
            closedDomain0 = newCubeLS(outerDim, innerDim, /*ctr=*/Vec3I(60, 0, 0), dx, false),
            closedDomain1 = newCubeLS(outerDim, innerDim, /*ctr=*/Vec3I(120, 0, 0), dx, false);

        // Union all three cubes into one grid.
        tools::csgUnion(*openDomain, *closedDomain0);
        tools::csgUnion(*openDomain, *closedDomain1);

        // Strictly speaking the solidBoundary level set should be rebuilt
        // (with tools::levelSetRebuild()) after the csgUnions to insure a proper
        // signed distance field, but we will forgo the rebuild in this example.
        solidBoundary = openDomain;
    }

    // Generate the source for the Poisson solver.
    // For a liquid simulation this will be the divergence of the velocity field
    // and will coincide with the liquid location.
    //
    // We activate by hand cells in distinct solution regions.

    FloatTree source(/*background=*/0.f);

    // The source is active in the union of the following "liquid" regions:

    // Fill the open box.
    const int N = 15;
    CoordBBox liquidInOpenDomain(Coord(-N, -N, -N), Coord(N, N, N));
    source.fill(liquidInOpenDomain, 0.f);

    // Totally fill closed box 0.
    CoordBBox liquidInClosedDomain0(Coord(-N, -N, -N), Coord(N, N, N));
    liquidInClosedDomain0.translate(Coord(60, 0, 0));
    source.fill(liquidInClosedDomain0, 0.f);

    // Half fill closed box 1.
    CoordBBox liquidInClosedDomain1(Coord(-N, -N, -N), Coord(N, N, 0));
    liquidInClosedDomain1.translate(Coord(120, 0, 0));
    source.fill(liquidInClosedDomain1, 0.f);

    // Compute the number of voxels in the well-posed region of the source.
    const Index64 expectedWellPosedVolume =
        liquidInOpenDomain.volume() + liquidInClosedDomain1.volume();

    // Generate a mask that defines the solution domain.
    // Inactive values of the source map to false and active values map to true.
    const BoolTree totalSourceDomain(source, /*inactive=*/false, /*active=*/true, TopologyCopy());

    // Extract the "interior regions" from the solid boundary.
    // The result will correspond to the the walls of the boxes unioned with inside of the full box.
    const BoolTree::ConstPtr interiorMask = tools::extractEnclosedRegion(
        solidBoundary->tree(), /*isovalue=*/float(0), &totalSourceDomain);

    // Identify the well-posed part of the problem.
    BoolTree wellPosedDomain(source, /*inactive=*/false, /*active=*/true, TopologyCopy());
    wellPosedDomain.topologyDifference(*interiorMask);
    EXPECT_EQ(expectedWellPosedVolume, wellPosedDomain.activeVoxelCount());

    // Solve the well-posed Poisson equation.

    const double epsilon = math::Delta<float>::value();
    math::pcg::State state = math::pcg::terminationDefaults<float>();
    state.iterations = 200;
    state.relativeError = state.absoluteError = epsilon;

    util::NullInterrupter interrupter;

    // Define boundary conditions that are consistent with solution = 0
    // at the liquid/air boundary and with a linear response with depth.
    LSBoundaryOp boundaryOp(solidBoundary->tree());

    // Compute the solution
    FloatTree::Ptr wellPosedSolutionP =
        tools::poisson::solveWithBoundaryConditionsAndPreconditioner<PreconditionerType>(
            source, wellPosedDomain, boundaryOp, state, interrupter, /*staggered=*/true);

    EXPECT_EQ(expectedWellPosedVolume, wellPosedSolutionP->activeVoxelCount());
    EXPECT_TRUE(state.success);
    EXPECT_TRUE(state.iterations < 68);

    // Verify that the solution is linear with depth.
    for (FloatTree::ValueOnCIter it = wellPosedSolutionP->cbeginValueOn(); it; ++it) {
        Index32 depth;
        if (liquidInOpenDomain.isInside(it.getCoord())) {
            depth = 1 + liquidInOpenDomain.max().z() - it.getCoord().z();
        } else {
            depth = 1 + liquidInClosedDomain1.max().z() - it.getCoord().z();
        }
        EXPECT_NEAR(double(depth), double(*it), /*tolerance=*/10.0 * epsilon);
    }

#if 0
    // Optionally, one could attempt to compute the solution in the enclosed regions.
    {
        // Identify the potentially ill-posed part of the problem.
        BoolTree illPosedDomain(source, /*inactive=*/false, /*active=*/true, TopologyCopy());
        illPosedDomain.topologyIntersection(source);

        // Solve the Poisson equation in the two unconnected regions.
        FloatTree::Ptr illPosedSoln =
            tools::poisson::solveWithBoundaryConditionsAndPreconditioner<PreconditionerType>(
                source, illPosedDomain, LSBoundaryOp(*solidBoundary->tree()),
                state, interrupter, /*staggered=*/true);
    }
#endif
}

using namespace openvdb;

// 0 Neumann pressure, meaning Dirichlet velocity on its normal face.
// 1 interior pressure dofs.
// 4 Dirichlet pressure. In this setup it's on the right. It means that it's not a solid collider or an open channel.
class SmokeSolver {
public:
    SmokeSolver(float const voxelSize) { init(voxelSize); }

    void init(float const vs)
    {
        mXform = math::Transform::createLinearTransform(mVoxelSize);

        int xDim = 3; int yDim = 15; int zDim = 17;
        mMinBBox = Vec3s(0.f, 0.f, 0.f);
        mMaxBBox = Vec3s(xDim * mVoxelSize, yDim * mVoxelSize, zDim * mVoxelSize);
        mMinIdx = mXform->worldToIndexNodeCentered(mMinBBox);
        mMaxIdx = mXform->worldToIndexNodeCentered(mMaxBBox);
        mMaxStaggered = mMaxIdx + math::Coord(1);

        initFlags();
        initInteriorPressure();
        initVCurr();
        initDivGrids();
    }

    // In the Flip Example class, this is VelocityBCCorrectionOp
    void applyDirichletVelocity() {
        auto flagsAcc = mFlags->getAccessor();
        auto vAcc = mVCurr->getAccessor();
        for (auto iter = mFlags->beginValueOn(); iter; ++iter) {
            math::Coord ijk = iter.getCoord();
            math::Coord im1jk = ijk.offsetBy(-1, 0, 0);
            math::Coord ijm1k = ijk.offsetBy(0, -1, 0);
            math::Coord ijkm1 = ijk.offsetBy(0, 0, -1);

            int flag = flagsAcc.getValue(ijk);
            int flagim1jk = flagsAcc.getValue(im1jk);
            int flagijm1k = flagsAcc.getValue(ijm1k);
            int flagijkm1 = flagsAcc.getValue(ijkm1);
            // I'm an interior pressure and I need to check if any of my neighbor is Neumann
            if (flag == 1)
            {
                if (flagim1jk == 0)
                {
                    auto cv = vAcc.getValue(ijk);
                    Vec3f newVel(0, cv[1], cv[2]);
                    vAcc.setValue(ijk, newVel);
                }

                if (flagijm1k == 0) {
                    auto cv = vAcc.getValue(ijk);
                    Vec3f newVel(cv[0], 0, cv[2]);
                    vAcc.setValue(ijk, newVel);

                }

                if (flagijkm1 == 0) {
                    auto cv = vAcc.getValue(ijk);
                    Vec3f newVel(cv[0], cv[1], 0);
                    vAcc.setValue(ijk, newVel);
                }

            } else if (flag == 0) { // I'm a Neumann pressure and I need if any of my Neighbor is interior
                if (flagim1jk == 1)
                {
                    auto cv = vAcc.getValue(ijk);
                    Vec3f newVel(0, cv[1], cv[2]);
                    vAcc.setValue(ijk, newVel);
                }

                if (flagijm1k == 1) {
                    auto cv = vAcc.getValue(ijk);
                    Vec3f newVel(cv[0], 0, cv[2]);
                    vAcc.setValue(ijk, newVel);
                }

                if (flagijkm1 == 1) {
                    auto cv = vAcc.getValue(ijk);
                    Vec3f newVel(cv[0], cv[1], 0);
                    vAcc.setValue(ijk, newVel);
                }
            }
        }
    }

    struct BoundaryOp {
        BoundaryOp(Int32Grid::ConstPtr flags,
                    Vec3SGrid::ConstPtr dirichletVelocity,
                    float const voxelSize) :
                    flags(flags),
                    dirichletVelocity(dirichletVelocity),
                    voxelSize(voxelSize) {}

        void operator()(const openvdb::Coord& ijk,
                        const openvdb::Coord& neighbor,
                        double& source,
                        double& diagonal) const
        {
            float const dirichletBC = 0.f;
            int flag = flags->tree().getValue(neighbor);
            bool isNeumannPressure = (flag == 0);
            bool isDirichletPressure = (flag == 4);
            auto vNgbr = Vec3s::zero(); //dirichletVelocity->tree().getValue(neighbor);

            // TODO: Double check this:
            if (isNeumannPressure) {
                double delta = 0.0;
                // Neumann pressure from bbox
                if (neighbor.x() + 1 == ijk.x() /* left x-face */) {
                    delta += vNgbr[0];
                }
                if (neighbor.x() - 1 == ijk.x() /* right x-face */) {
                    delta -= vNgbr[0];
                }
                if (neighbor.y() + 1 == ijk.y() /* bottom y-face */) {
                    delta += vNgbr[1];
                }
                if (neighbor.y() - 1 == ijk.y() /* top y-face */) {
                    delta -= vNgbr[1];
                }
                if (neighbor.z() + 1 == ijk.z() /* back z-face */) {
                    delta += vNgbr[2];
                }
                if (neighbor.z() - 1 == ijk.z() /* front z-face */) {
                    delta -= vNgbr[2];
                }
                // Note: in the SOP_OpenVDB_Remove_Divergence, we need to multiply
                // this by 0.5, because the gradient that's used is using
                // central-differences in a collocated grid, instead of the staggered one.
                source += delta / voxelSize;
            } else if (isDirichletPressure) {
                diagonal -= 1.0;
                source -= dirichletBC;
#if 0 // supposedly the same as the two lines above--checked on Friday.
                // Dirichlet pressure
                if (neighbor.x() + 1 == ijk.x() /* left x-face */) {
                    diagonal -= 1.0;
                    source -= dirichletBC;
                }
                else if (neighbor.x() - 1 == ijk.x() /* right x-face */) {
                    diagonal -= 1.0;
                    source -= dirichletBC;
                }
                else if (neighbor.y() + 1 == ijk.y() /* bottom y-face */) {
                    diagonal -= 1.0;
                    source -= dirichletBC;
                }
                else if (neighbor.y() - 1 == ijk.y() /* top y-face */) {
                    diagonal -= 1.0;
                    source -= dirichletBC;
                }
                else if (neighbor.z() + 1 == ijk.z() /* back z-face */) {
                    diagonal -= 1.0;
                    source -= dirichletBC;
                }
                else if (neighbor.z() - 1 == ijk.z() /* front z-face */) {
                    diagonal -= 1.0;
                    source -= dirichletBC;
                }
#endif
            }
        }

        Int32Grid::ConstPtr flags;
        Vec3SGrid::ConstPtr dirichletVelocity;
        float voxelSize;
    };

    void subtractPressureGradFromVel() {
        auto vCurrAcc = mVCurr->getAccessor();
        auto pressureAcc = mPressure->getConstAccessor();
        auto flagsAcc = mFlags->getConstAccessor();
        for (auto iter = mVCurr->beginValueOn(); iter; ++iter) {
                math::Coord ijk = iter.getCoord();
                math::Coord im1jk = ijk.offsetBy(-1, 0, 0);
                math::Coord ijm1k = ijk.offsetBy(0, -1, 0);
                math::Coord ijkm1 = ijk.offsetBy(0, 0, -1);

            // Only updates velocity if it is a face of fluid cell
            if (flagsAcc.getValue(ijk) == 1 ||
                flagsAcc.getValue(im1jk) == 1 ||
                flagsAcc.getValue(ijm1k) == 1 ||
                flagsAcc.getValue(ijkm1) == 1) {
                Vec3s gradijk;
                gradijk[0] = pressureAcc.getValue(ijk) - pressureAcc.getValue(ijk.offsetBy(-1, 0, 0));
                gradijk[1] = pressureAcc.getValue(ijk) - pressureAcc.getValue(ijk.offsetBy(0, -1, 0));
                gradijk[2] = pressureAcc.getValue(ijk) - pressureAcc.getValue(ijk.offsetBy(0, 0, -1));
                auto val = vCurrAcc.getValue(ijk) - gradijk * mVoxelSize;
                vCurrAcc.setValue(ijk, val);
            }
        }

        applyDirichletVelocity(); // VERY IMPORTANT
    }

    void pressureProjection() {
        using TreeType = FloatTree;
        using ValueType = TreeType::ValueType;
        using PCT = openvdb::math::pcg::JacobiPreconditioner<openvdb::tools::poisson::LaplacianMatrix>;

        BoundaryOp bop(mFlags, mVCurr, mVoxelSize);
        util::NullInterrupter interrupter;

        const double epsilon = math::Delta<ValueType>::value();

        mState = math::pcg::terminationDefaults<ValueType>();
        mState.iterations = 100;
        mState.relativeError = mState.absoluteError = epsilon;
        FloatTree::Ptr fluidPressure = tools::poisson::solveWithBoundaryConditionsAndPreconditioner<PCT>(
            mDivBefore->tree(), mInteriorPressure->tree(), bop, mState, interrupter, /*staggered=*/true);

        FloatGrid::Ptr fluidPressureGrid = FloatGrid::create(fluidPressure);
        fluidPressureGrid->setTransform(mXform);
        mPressure = fluidPressureGrid->copy();
        mPressure->setName("pressure");
    }

    void render()
    {
        if (mVERBOSE) printRelevantVelocity("velocity init");

        float divBefore = computeDivergence(mDivBefore, mVCurr, "before");
        if (mVERBOSE) printGrid(*mDivBefore);

        // Make the velocity divergence free by solving Poisson Equation and subtracting the pressure gradient
        pressureProjection();
        subtractPressureGradFromVel();

        float divAfter = computeDivergence(mDivAfter, mVCurr, "after");
        if (mVERBOSE) printGrid(*mDivAfter);

        writeVDBsDebug(1);
    }


    template<class GridType>
    typename GridType::Ptr
    initGridBgAndName(typename GridType::ValueType background, std::string name)
    {
        typename GridType::Ptr grid = GridType::create(background);
        grid->setTransform(mXform);
        grid->setName(name);
        return grid;
    }

    template<class GridType>
    void printGrid(const GridType& grid, std::string nameFromUser = "") {
        using ValueType = typename GridType::ValueType;
        auto name = nameFromUser != "" ? nameFromUser : grid.getName();
        std::cout << "printGrid::Printing grid " << name << std::endl;
        auto acc = grid.getAccessor();
        for (auto iter = grid.beginValueOn(); iter; ++iter) {
            math::Coord ijk = iter.getCoord();
            std::cout << "val" << ijk << " = " << acc.getValue(ijk) << std::endl;
        }
        std::cout << std::endl;
    }

    void printRelevantVelocity(std::string nameFromUser = "") {
        std::cout << "printRelevantVelocity::printing " << nameFromUser << std::endl;
        auto flagsAcc = mFlags->getAccessor();
        auto vAcc = mVCurr->getAccessor();
        for (auto iter = mFlags->beginValueOn(); iter; ++iter) {
            math::Coord ijk = iter.getCoord();
            math::Coord im1jk = ijk.offsetBy(-1, 0, 0);
            math::Coord ijm1k = ijk.offsetBy(0, -1, 0);
            math::Coord ijkm1 = ijk.offsetBy(0, 0, -1);

            int flag = flagsAcc.getValue(ijk);
            int flagim1jk = flagsAcc.getValue(im1jk);
            int flagijm1k = flagsAcc.getValue(ijm1k);
            int flagijkm1 = flagsAcc.getValue(ijkm1);

            if (flag == 1) {
                std::cout << "vel" << ijk << " = " << vAcc.getValue(ijk) << std::endl;
            } else {
                if (flagim1jk == 1 || flagijm1k == 1 || flagijkm1 == 1) {
                    std::cout << "vel" << ijk << " = " << vAcc.getValue(ijk) << std::endl;
                }
            }
        }
    }

    void initFlags()
    {
        mFlags = initGridBgAndName<Int32Grid>(0, "flags");
        mFlags->denseFill(CoordBBox(mMinIdx, mMaxIdx), /* value = */ 1, /* active = */ true);

        auto flagsAcc = mFlags->getAccessor();
        for (auto iter = mFlags->beginValueOn(); iter; ++iter) {
            math::Coord ijk = iter.getCoord();

            if (ijk[0] == mMaxIdx[0]) {
                flagsAcc.setValue(ijk, 4); // Dirichlet
            }
            if (ijk[0] == mMinIdx[0] /* left face */ ||
                ijk[1] == mMinIdx[1] /* bottom face */ ||
                ijk[1] == mMaxIdx[1] /* top face */ ||
                ijk[2] == mMinIdx[2] /* back face */ ||
                ijk[2] == mMaxIdx[2] /* front face */) {
                flagsAcc.setValue(ijk, 0); // Neumann
            }
        }
    }

    void initDivGrids() {
        mDivBefore = initGridBgAndName<FloatGrid>(0.f, "div_before");
        mDivAfter = initGridBgAndName<FloatGrid>(0.f, "div_after");
    }

    float computeLInfinity(const FloatGrid& grid) {
        float ret = 0.f;
        auto acc = grid.getConstAccessor();
        for (auto iter = grid.beginValueOn(); iter; ++iter) {
            math::Coord ijk = iter.getCoord();
            auto val = acc.getValue(ijk);
            if (std::abs(val) > std::abs(ret)) {
                ret = val;
            }
        }
        return ret;
    }

    float computeDivergence(FloatGrid::Ptr& divGrid, const Vec3SGrid::Ptr vecGrid, const std::string& suffix) {
        divGrid = tools::divergence(*vecGrid);
        divGrid->tree().topologyIntersection(mInteriorPressure->tree());
        float div = computeLInfinity(*divGrid);
        std::cout << "Divergence " << suffix.c_str() << " = " << div << std::endl;
        return div;
    }

    void initVCurr()
    {
        mVCurr = initGridBgAndName<Vec3SGrid>(Vec3s::zero(), "vel_curr");
        mVCurr->setGridClass(GRID_STAGGERED);
        mVCurr->denseFill(CoordBBox(mMinIdx, mMaxStaggered), /* value = */ Vec3s(0.f, 0.f, 0.f), /* active = */ true);

        auto flagsAcc = mFlags->getConstAccessor();
        auto velAcc = mVCurr->getAccessor();
        const float hv = .5f * mXform->voxelSize()[0]; // half of voxel size
        for (auto iter = mVCurr->beginValueOn(); iter; ++iter) {
            auto ijk = iter.getCoord();
            Vec3f center = mXform->indexToWorld(ijk);

            float x = center[0] - hv;
            float y = center[1] - hv;
            float z = center[2] - hv;
            Vec3s val(x * x, y * y, z *z);
            velAcc.setValue(ijk, val);
        }

        applyDirichletVelocity(); // VERY IMPORTANT
    }

    void initInteriorPressure()
    {
        mInteriorPressure = initGridBgAndName<BoolGrid>(false, "interior_pressure");
        mInteriorPressure->denseFill(CoordBBox(mMinIdx, mMaxIdx), /* value = */ true, /* active = */ true);

        auto flagsAcc = mFlags->getConstAccessor();
        for (auto iter = mInteriorPressure->beginValueOn(); iter; ++iter) {
            math::Coord ijk = iter.getCoord();
            if (flagsAcc.getValue(ijk) != 1) {
                iter.setValueOff();
            }
        }
    }

    void writeVDBsDebug(int const frame) {
        std::ostringstream ss;
        ss << "INIT_DEBUG" << std::setw(3) << std::setfill('0') << frame << ".vdb";
        std::string fileName(ss.str());
        io::File file(fileName.c_str());

        openvdb::GridPtrVec grids;
        grids.push_back(mFlags);
        grids.push_back(mInteriorPressure);
        grids.push_back(mVCurr);
        grids.push_back(mDivBefore);
        grids.push_back(mDivAfter);
        grids.push_back(mPressure);

        file.write(grids);
        file.close();
    }

    bool mVERBOSE = false;

    float mVoxelSize = 0.1f;
    math::Transform::Ptr mXform;

    math::pcg::State mState;

    Vec3s mMaxBBox, mMinBBox;
    Coord mMinIdx, mMaxIdx;
    Coord mMaxStaggered;

    Int32Grid::Ptr mFlags;
    BoolGrid::Ptr mInteriorPressure;
    Vec3SGrid::Ptr mVCurr;
    FloatGrid::Ptr mPressure;
    FloatGrid::Ptr mDivBefore;
    FloatGrid::Ptr mDivAfter;
};


TEST_F(TestPoissonSolver, testRemoveDivergence)
{
    using namespace openvdb;

    SmokeSolver smoke(0.1f);

    float divBefore = smoke.computeDivergence(smoke.mDivBefore, smoke.mVCurr, "before");

    // Make the velocity divergence free by solving Poisson Equation and subtracting the pressure gradient
    smoke.pressureProjection();
    smoke.subtractPressureGradFromVel();

    EXPECT_TRUE(smoke.mPressure);
    EXPECT_EQ(smoke.mState.success, 1);
    EXPECT_EQ(smoke.mState.iterations, 28);
    EXPECT_TRUE(smoke.mState.relativeError < 1.e-5f);
    EXPECT_TRUE(smoke.mState.absoluteError < 1.e-3f);

    float divAfter = smoke.computeDivergence(smoke.mDivAfter, smoke.mVCurr, "after");
    EXPECT_TRUE(divAfter < 1.e-3f);
    smoke.writeVDBsDebug(1 /* frame */);
}
