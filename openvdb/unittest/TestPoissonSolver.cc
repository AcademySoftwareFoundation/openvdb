///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
//#include <openvdb/math/Math.h> // for math::isApproxEqual()
#include <openvdb/math/ConjGradient.h> // for JacobiPreconditioner
#include <openvdb/tools/Composite.h> // for csgDifference/Union/Intersection
#include <openvdb/tools/LevelSetSphere.h> // for tools::createLevelSetSphere()
#include <openvdb/tools/LevelSetUtil.h> // for tools::sdfToFogVolume()
#include <openvdb/tools/MeshToVolume.h> // for createLevelSetBox()
#include <openvdb/tools/Morphology.h> // for tools::erodeVoxels()
#include <openvdb/tools/PoissonSolver.h>
#include <boost/math/constants/constants.hpp> // for boost::math::constants::pi
#include <cmath>

/// @authors D.J. Hill, Peter Cucka

class TestPoissonSolver: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestPoissonSolver);
    CPPUNIT_TEST(testIndexTree);
    CPPUNIT_TEST(testTreeToVectorToTree);
    CPPUNIT_TEST(testLaplacian);
    CPPUNIT_TEST(testSolve);
    CPPUNIT_TEST(testSolveWithBoundaryConditions);
    CPPUNIT_TEST(testSolveWithSegmentDomain);
    CPPUNIT_TEST_SUITE_END();

    void testIndexTree();
    void testTreeToVectorToTree();
    void testLaplacian();
    void testSolve();
    void testSolveWithBoundaryConditions();
    void testSolveWithSegmentDomain();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestPoissonSolver);


////////////////////////////////////////


void
TestPoissonSolver::testIndexTree()
{
    using namespace openvdb;
    using tools::poisson::VIndex;

    typedef FloatTree::ValueConverter<VIndex>::Type VIdxTree;
    typedef VIdxTree::LeafNodeType LeafNodeType;

    VIdxTree tree;
    /// @todo populate tree
    tree::LeafManager<const VIdxTree> leafManager(tree);

    VIndex testOffset = 0;
    for (size_t n = 0, N = leafManager.leafCount(); n < N; ++n) {
        const LeafNodeType& leaf = leafManager.leaf(n);
        for (LeafNodeType::ValueOnCIter it = leaf.cbeginValueOn(); it; ++it, testOffset++) {
            CPPUNIT_ASSERT_EQUAL(testOffset, *it);
        }
    }

    //if (testOffset != VIndex(tree.activeVoxelCount())) {
    //    std::cout << "--Testing offsetmap - "
    //              << testOffset<<" != "
    //              << tree.activeVoxelCount()
    //              << " has active tile count "
    //              << tree.activeTileCount()<<std::endl;
    //}

    CPPUNIT_ASSERT_EQUAL(VIndex(tree.activeVoxelCount()), testOffset);
}


void
TestPoissonSolver::testTreeToVectorToTree()
{
    using namespace openvdb;
    using tools::poisson::VIndex;

    typedef FloatTree::ValueConverter<VIndex>::Type VIdxTree;

    FloatGrid::Ptr sphere = tools::createLevelSetSphere<FloatGrid>(
        /*radius=*/10.f, /*center=*/Vec3f(0.f), /*voxelSize=*/0.25f);
    tools::sdfToFogVolume(*sphere);
    FloatTree& inputTree = sphere->tree();

    const Index64 numVoxels = inputTree.activeVoxelCount();

    // Generate an index tree.
    VIdxTree::Ptr indexTree = tools::poisson::createIndexTree(inputTree);
    CPPUNIT_ASSERT(bool(indexTree));

    // Copy the values of the active voxels of the tree into a vector.
    math::pcg::VectorS::Ptr vec =
        tools::poisson::createVectorFromTree<float>(inputTree, *indexTree);
    CPPUNIT_ASSERT_EQUAL(math::pcg::SizeType(numVoxels), vec->size());

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
            CPPUNIT_ASSERT_DOUBLES_EQUAL(inputAcc.getValue(ijk), *it, /*tolerance=*/1.0e-6);
        }
    }
}


void
TestPoissonSolver::testLaplacian()
{
    using namespace openvdb;
    using tools::poisson::VIndex;

    typedef FloatTree::ValueConverter<VIndex>::Type VIdxTree;

    // For two different problem sizes, N = 8 and N = 20...
    for (int N = 8; N <= 20; N += 12) {
        // Construct an N x N x N volume in which the value of voxel (i, j, k)
        // is sin(i) * sin(j) * sin(k), using a voxel spacing of pi / N.
        const double delta = boost::math::constants::pi<double>() / N;
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
        CPPUNIT_ASSERT(bool(indexTree));

        // Copy the values of the active voxels of the tree into a vector.
        math::pcg::VectorS::Ptr source =
            tools::poisson::createVectorFromTree<float>(inputTree, *indexTree);
        CPPUNIT_ASSERT_EQUAL(math::pcg::SizeType(numVoxels), source->size());

        // Create a mask of the interior voxels of the source tree.
        BoolTree interiorMask(/*background=*/false);
        interiorMask.fill(CoordBBox(Coord(2), Coord(N-2)), /*value=*/true, /*active=*/true);

        // Compute the Laplacian of the source:
        //     D^2 sin(i) * sin(j) * sin(k) = -3 sin(i) * sin(j) * sin(k)
        tools::poisson::LaplacianMatrix::Ptr laplacian =
            tools::poisson::createISLaplacian(*indexTree, interiorMask);
        laplacian->scale(1.0 / (delta * delta)); // account for voxel spacing
        CPPUNIT_ASSERT_EQUAL(math::pcg::SizeType(numVoxels), laplacian->size());

        math::pcg::VectorS result(source->size());
        laplacian->vectorMultiply(*source, result);

        // Dividing the result by the source should produce a vector of uniform value -3.
        // Due to finite differencing, the actual ratio will be somewhat different, though.
        const math::pcg::VectorS& src = *source;
        const float expected = // compute the expected ratio using one of the corner voxels
            float((3.0 * src[1] - 6.0 * src[0]) / (delta * delta * src[0]));
        for (math::pcg::SizeType n = 0; n < result.size(); ++n) {
            result[n] /= src[n];
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, result[n], /*tolerance=*/1.0e-4);
        }
    }
}


void
TestPoissonSolver::testSolve()
{
    using namespace openvdb;

    FloatGrid::Ptr sphere = tools::createLevelSetSphere<FloatGrid>(
        /*radius=*/10.f, /*center=*/Vec3f(0.f), /*voxelSize=*/0.25f);
    tools::sdfToFogVolume(*sphere);

    math::pcg::State result = math::pcg::terminationDefaults<float>();
    result.iterations = 100;
    result.relativeError = result.absoluteError = 1.0e-4;

    FloatTree::Ptr outTree = tools::poisson::solve(sphere->tree(), result);

    CPPUNIT_ASSERT(result.success);
    CPPUNIT_ASSERT(result.iterations < 60);
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

    typedef typename TreeType::ValueType ValueType;

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
    // (0,-N-1,0) +------+ (N,-N-1,N)
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
        source, BoundaryOp(), state, interrupter);

    CPPUNIT_ASSERT(state.success);
    CPPUNIT_ASSERT(state.iterations < 60);

    // Verify that P = -y throughout the solution space.
    for (typename TreeType::ValueOnCIter it = solution->cbeginValueOn(); it; ++it) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            double(-it.getCoord().y()), double(*it), /*tolerance=*/10.0 * epsilon);
    }
}

} // unnamed namespace


void
TestPoissonSolver::testSolveWithBoundaryConditions()
{
    doTestSolveWithBoundaryConditions<openvdb::FloatTree>();
    doTestSolveWithBoundaryConditions<openvdb::DoubleTree>();
}


namespace {
    
    openvdb::FloatGrid::Ptr generateCubeLS(const int outer_length, // in voxels
                                           const int inner_length, // in voxels
                                           const openvdb::Vec3I& centerIS, // in index space
                                           const float dx, // grid spacing
                                           bool open_top)
    {

        using namespace openvdb;

        typedef math::BBox<Vec3f> BBox;
    
        // World space dimensions and center for this box
        const float outerWS = dx * float(outer_length);
        const float innerWS = dx * float(inner_length); 
        Vec3f centerWS(centerIS);
        centerWS *= dx;
    
        
  
        // Construct world space bounding boxes
        BBox outerBBox(Vec3f(-outerWS /2 , -outerWS / 2, -outerWS /2), Vec3f(outerWS / 2, outerWS / 2, outerWS / 2));
        BBox innerBBox;
        if ( open_top) {
            innerBBox = BBox(Vec3f(-innerWS /2 , -innerWS / 2, -innerWS /2), Vec3f(innerWS / 2, innerWS / 2, outerWS ));
        } else {
            innerBBox = BBox(Vec3f(-innerWS /2 , -innerWS / 2, -innerWS /2), Vec3f(innerWS / 2, innerWS / 2, innerWS / 2));
        }
        outerBBox.translate(centerWS);
        innerBBox.translate(centerWS);
        

        math::Transform::Ptr xform = math::Transform::createLinearTransform(dx);
        FloatGrid::Ptr cubeLS = tools::createLevelSetBox<FloatGrid>( outerBBox, *xform);
        FloatGrid::Ptr inside = tools::createLevelSetBox<FloatGrid>( innerBBox, *xform);
        tools::csgDifference(*cubeLS, *inside);

        return cubeLS;
    }
  
    
    class LSBoundaryOp {
    public:
        LSBoundaryOp(const openvdb::FloatTree& lsTree) : mLS(&lsTree) {}
        LSBoundaryOp(const LSBoundaryOp& other) :mLS(other.mLS) {}
        
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
}// unnamed namespace        

void 
TestPoissonSolver::testSolveWithSegmentDomain()
{

    using namespace openvdb;

    typedef math::pcg::IncompleteCholeskyPreconditioner<tools::poisson::LaplacianMatrix>    PreconditionerType;

    // In fluid simulations, incomprehensibility is enforced by the pressure, which in turn is computed as a solution of a Poisson equation.
    // Often procedural animation of objects (e.g. characters) interacting with liquid will result in boundary conditions that describe
    // multiple disjoint regions: regions of free surface flow and regions of of trapped fluid. It is this second type of region 
    // for which there may be no consistent pressure (e.g. a shrinking water tight region filled with incompressible liquid).

    // This unittest demonstrates how to separate the well-posed problem of a liquid with a free surface from the possibly ill-posed 
    // fully enclosed liquid regions by using a levelset and topological tools.


    // For simplicity sake, the physical boundaries are idealized as three non-overlapping cubes.  
    // One with an open top, and two that are fully closed.  
    // 
    // All three containing incompressible liquid(x) and one of the closed cubes one will be partially filled so
    // that two of the liquid regions have a free surface (Dirichlet BC on one side) 
    // while the totally filled cube would have no free surface (Neumann BCs on all sides).

    //
    //                                    ________________              ________________
    //        __            __           |   __________   |            |   __________   |
    //       |  |x x x x x |  |          |  |          |  |            |  |x x x x x |  |  
    //       |  |x x x x x |  |          |  |x x x x x |  |            |  |x x x x x |  |  
    //       |  |x x x x x |  |          |  |x x x x x |  |            |  |x x x x x |  |  
    //       |   ——————————   |          |   ——————————   |            |   ——————————   |  
    //       |________________|          |________________|            |________________|  
    //

    // The first two regions are clearly well-posed while the third region may have no solution (or multiple solutions). 

    // -D.J.Hill

    // Grid spacing

    const float dx = 0.05f;


    // Construct the solid boundaries in a single grid.

    FloatGrid::Ptr  solidBoundary;
    { 
        const int outer_dimension = 41;
        const int inner_dimension = 31;
        FloatGrid::Ptr openDomain    = generateCubeLS(outer_dimension, inner_dimension, Vec3I(0, 0, 0) /*center*/, dx, true /*=open top*/);
        FloatGrid::Ptr closedDomain0 = generateCubeLS(outer_dimension, inner_dimension, Vec3I(60, 0, 0) /*center*/, dx, false /*=open top*/);
        FloatGrid::Ptr closedDomain1 = generateCubeLS(outer_dimension, inner_dimension, Vec3I(120, 0, 0) /*center*/, dx, false /*=open top*/);
        
        // // Union the cubes into the opencube grid
        
        tools::csgUnion(*openDomain, *closedDomain0);
        tools::csgUnion(*openDomain, *closedDomain1);
        
        // rename.
        
        solidBoundary = openDomain;

        // Strictly speaking the soulidBoundary level set should be rebuilt (tools::levelSetRebuild) 
        // after csgUnions to insure a proper signed distance field but will will forgo it in this example
  
    }
    
    
    // Generate the source for the Poisson solver. 
    // For a liquid simulation this will be the divergence of the velocity field and will coincide with the liquid location.
    // 
    // We activate by hand cells in distinct solution regions.
    
    FloatTree source(0.f /*background*/);
    
    const int N = 15;
    // The source is active the union of the following "liquid" regions.
    
    // Fill the open box
    CoordBBox    liquidInOpenDomain(Coord(-N, -N, -N), Coord(N, N, N));
    source.fill(liquidInOpenDomain, 0.f);
    
    // Totally fill the closed box 0
    CoordBBox liquidInClosedDomain0(Coord(-N, -N, -N), Coord(N, N, N));   liquidInClosedDomain0.translate(Coord(60, 0, 0));
    source.fill(liquidInClosedDomain0, 0.f);

    // Half fill the closed box 1
    CoordBBox liquidInClosedDomain1(Coord(-N, -N, -N), Coord(N, N, 0));  liquidInClosedDomain1.translate(Coord(120, 0, 0));
    source.fill(liquidInClosedDomain1, 0.f);
    
    // The number of voxels in the part of the source that will correspond to the well posed region
    
    Index64 expectedWellPosedVolume = liquidInOpenDomain.volume() + liquidInClosedDomain1.volume();
    

    // Generate a mask that defines the solution domain

    const BoolTree totalSourceDomain( source, false/*background*/, true/*active*/, TopologyCopy());

    // Extract the "interior regions" from the solidBoundary.  
    // The result will correspond to the the walls of the boxes union-ed with inside of the full box.
    
    BoolTree::Ptr interiorMask = tools::extractEnclosedRegion(solidBoundary->tree(), float(0) /*isovalue*/, &totalSourceDomain);
                                            
    
    // Identify the  well-posed part of the problem 
    
    BoolTree wellPosedDomain( source, false/*inactive*/, true/*active*/, TopologyCopy());
    wellPosedDomain.topologyDifference(*interiorMask);
    CPPUNIT_ASSERT(wellPosedDomain.activeVoxelCount() == expectedWellPosedVolume );
   

    // Solve well posed Poisson equation
    const double epsilon = math::Delta<float>::value();
    math::pcg::State state = math::pcg::terminationDefaults<float>();
    state.iterations = 200;
    state.relativeError = state.absoluteError = epsilon;

    util::NullInterrupter interrupter;

    // Boundary conditions that are consistent with solution = 0 at liquid air boundary
    // and a linear response with depth.  

    LSBoundaryOp boundaryOp(solidBoundary->tree());
    

    // Compute the solution

    FloatTree::Ptr wellPosedSolutionP   = 
        tools::poisson::solveWithBoundaryConditionsAndPreconditioner<PreconditionerType>(source,
                                                                                         wellPosedDomain,
                                                                                         boundaryOp, 
                                                                                         state, interrupter);
    
    
    CPPUNIT_ASSERT(wellPosedSolutionP->activeVoxelCount() == expectedWellPosedVolume );
    CPPUNIT_ASSERT(state.success);
    CPPUNIT_ASSERT(state.iterations < 68);
    


    // Verify that solution is linear with depth.

    for (FloatTree::ValueOnCIter it = wellPosedSolutionP->cbeginValueOn(); it; ++it) {
        Index32 depth;
        if (liquidInOpenDomain.isInside(it.getCoord())) {
            
           depth = 1 + liquidInOpenDomain.max().z() - it.getCoord().z();
           
        } else {

            depth = 1 + liquidInClosedDomain1.max().z() - it.getCoord().z();
        }
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(depth), double(*it), /*tolerance=*/10.0 * epsilon);
    }


#if 0   // optionally, one could attempt to compute the solution in the enclosed regions.
    {

        // Identify the potentially ill-posed part of the problem 
        
        BoolTree illPosedDomain( source, false/*inactive*/, true/*active*/, TopologyCopy());
        illPosedDomain.topologyIntersection(source);
                
           
        // Solve for Poisson equation in the two unconnected regions
        FloatTree::Ptr illPosedSoln = 
            tools::poisson::solveWithBoundaryConditionsAndPreconditioner<PreconditionerType>(source, illPosedDomain, 
                                                                                             LSBoundaryOp(*solidBoundary->tree()), 
                                                                                             state, interrupt); 
    }
#endif

}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
