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
//
/// @file SOP_OpenVDB_Remove_Divergence.cc
///
/// @author FX R&D OpenVDB team


#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>

#include <openvdb/math/ConjGradient.h> // for JacobiPreconditioner
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/PoissonSolver.h>
#include <openvdb/tools/ChangeBackground.h>

#include <UT/UT_Interrupt.h>
#include <GU/GU_Detail.h>
#include <PRM/PRM_Parm.h>
#include <GA/GA_Handle.h>
#include <GA/GA_PageIterator.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

////////////////////////////////////////

// Local Utility Methods

namespace {


template<typename TreeType>
struct CorrectVelocityOp
{
    typedef typename TreeType::LeafNodeType LeafNodeType;
    typedef typename TreeType::ValueType    ValueType;

    CorrectVelocityOp(LeafNodeType** velocityNodes, const LeafNodeType** gradientOfPressureNodes, double dx)
        : mVelocityNodes(velocityNodes), mGradientOfPressureNodes(gradientOfPressureNodes), mVoxelSize(dx)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        typedef typename ValueType::value_type ElementType;
        const ElementType scale = ElementType(mVoxelSize * mVoxelSize);

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            LeafNodeType& velocityNode = *mVelocityNodes[n];
            ValueType* velocityData = velocityNode.buffer().data();

            const ValueType* gradientOfPressureData = mGradientOfPressureNodes[n]->buffer().data();

            for (typename LeafNodeType::ValueOnIter it = velocityNode.beginValueOn(); it; ++it) {
                const openvdb::Index pos = it.pos();
                velocityData[pos] -= scale * gradientOfPressureData[pos];
            }
        }
    }

    LeafNodeType       * const * const mVelocityNodes;
    LeafNodeType const * const * const mGradientOfPressureNodes;
    double                       const mVoxelSize;
}; // class CorrectVelocityOp


/// Constant boundary condition functor
struct DirichletOp {
    inline void operator()(const openvdb::Coord&,
        const openvdb::Coord&, double&, double& diag) const { diag -= 1; }
};


template<typename VectorGridType>
inline bool
removeDivergence(VectorGridType& velocityGrid, const int iterations, hvdb::Interrupter& interrupter)
{
    typedef typename VectorGridType::TreeType       VectorTreeType;
    typedef typename VectorTreeType::LeafNodeType   VectorLeafNodeType;
    typedef typename VectorGridType::ValueType      VectorType;
    typedef typename VectorType::ValueType          VectorElementType;

    typedef typename VectorGridType::template ValueConverter<VectorElementType>::Type   ScalarGrid;
    typedef typename ScalarGrid::TreeType                                               ScalarTree;

    openvdb::tools::Divergence<VectorGridType> divergenceOp(velocityGrid);
    typename ScalarGrid::Ptr divGrid = divergenceOp.process();

    openvdb::math::pcg::State state = openvdb::math::pcg::terminationDefaults<VectorElementType>();
    state.iterations = iterations;
    state.relativeError = state.absoluteError = openvdb::math::Delta<VectorElementType>::value();

    typedef openvdb::math::pcg::JacobiPreconditioner<openvdb::tools::poisson::LaplacianMatrix> PCT;

    typename ScalarTree::Ptr pressure =
        openvdb::tools::poisson::solveWithBoundaryConditionsAndPreconditioner<PCT>(
            divGrid->tree(), DirichletOp(), state, interrupter);

    typename ScalarGrid::Ptr pressureGrid = ScalarGrid::create(pressure);
    pressureGrid->setTransform(velocityGrid.transform().copy());

    openvdb::tools::Gradient<ScalarGrid> gradientOp(*pressureGrid);
    typename VectorGridType::Ptr gradientOfPressure = gradientOp.process();

    {
        std::vector<VectorLeafNodeType*> velocityNodes;
        velocityGrid.tree().getNodes(velocityNodes);

        std::vector<const VectorLeafNodeType*> gradientNodes;
        gradientNodes.reserve(velocityNodes.size());
        gradientOfPressure->tree().getNodes(gradientNodes);

        const double dx = velocityGrid.transform().voxelSize()[0];

        tbb::parallel_for(tbb::blocked_range<size_t>(0, velocityNodes.size()),
            CorrectVelocityOp<VectorTreeType>(&velocityNodes[0], &gradientNodes[0], dx));
    }

    return state.success;
}


} // unnamed namespace


////////////////////////////////////////

// SOP Implementation

struct SOP_OpenVDB_Remove_Divergence: public hvdb::SOP_NodeVDB
{
    SOP_OpenVDB_Remove_Divergence(OP_Network*, const char* name, OP_Operator*);
    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
}; // SOP_OpenVDB_Remove_Divergence


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify vector grids to process")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    parms.add(hutil::ParmFactory(PRM_INT_J, "iterations", "Iterations")
        .setDefault(50)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 100));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Remove Divergence",
        SOP_OpenVDB_Remove_Divergence::factory, parms, *table)
        .addInput("VDB Grid");
}

bool
SOP_OpenVDB_Remove_Divergence::updateParmsFlags()
{
    bool changed = false;
    return changed;
}


OP_Node*
SOP_OpenVDB_Remove_Divergence::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Remove_Divergence(net, name, op);
}


SOP_OpenVDB_Remove_Divergence::SOP_OpenVDB_Remove_Divergence(
    OP_Network* net, const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDB(net, name, op)
{
}

OP_ERROR
SOP_OpenVDB_Remove_Divergence::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        duplicateSourceStealable(0, context);

        const fpreal time = context.getTime();

        hvdb::Interrupter boss("Removing Divergence");

        const int iterations = evalInt("iterations", 0, time);

        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        bool processedVDB = false;

        for (hvdb::VdbPrimIterator vdbIt(gdp, group); vdbIt; ++vdbIt) {

            if (boss.wasInterrupted()) break;

            if (vdbIt->getGrid().type() == openvdb::Vec3fGrid::gridType()) {

                processedVDB = true;

                vdbIt->makeGridUnique();

                openvdb::Vec3fGrid& grid = static_cast<openvdb::Vec3fGrid&>(vdbIt->getGrid());

                if (!removeDivergence(grid, iterations, boss) && !boss.wasInterrupted()) {
                    const std::string msg = grid.getName() + " did not fully converge.";
                    addWarning(SOP_MESSAGE, msg.c_str());
                }
            }
        }

        if (!processedVDB && !boss.wasInterrupted()) {
            addWarning(SOP_MESSAGE, "No Vec3f VDBs found.");
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )


