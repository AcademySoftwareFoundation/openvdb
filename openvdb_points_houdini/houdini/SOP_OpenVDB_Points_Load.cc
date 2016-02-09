///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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
/// @file SOP_OpenVDB_Points_Load.cc
///
/// @author Dan Bailey
///
/// @brief Explicitly loads OpenVDB points that are delay-loaded.


#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>

#include "SOP_NodeVDBPoints.h"

#include <houdini_utils/geometry.h>
#include <houdini_utils/ParmFactory.h>

using namespace openvdb;
using namespace openvdb::tools;
using namespace openvdb::math;

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


class SOP_OpenVDB_Points_Load: public hvdb::SOP_NodeVDBPoints
{
public:
    SOP_OpenVDB_Points_Load(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Points_Load() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:

    virtual OP_ERROR cookMySop(OP_Context&);

private:
    hvdb::Interrupter mBoss;
}; // class SOP_OpenVDB_Points_Load



////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    points::initialize();

    if (table == NULL) return;

    hutil::ParmList parms;

    //////////
    // Register this operator.

    hvdb::OpenVDBOpFactory("OpenVDB Points Load",
        SOP_OpenVDB_Points_Load::factory, parms, *table)
        .addInput("Points to Load");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Points_Load::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Points_Load(net, name, op);
}


SOP_OpenVDB_Points_Load::SOP_OpenVDB_Points_Load(OP_Network* net,
    const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDBPoints(net, name, op)
{
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Points_Load::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSourceStealable(0, context);

        UT_AutoInterrupt progress("Processing Points Load");

        if (gdp == NULL) return error();

        hvdb::VdbPrimIterator vdbIt(gdp);

        // Handle no vdbs
        if (!vdbIt) {
            addError(SOP_MESSAGE, "No VDBs found.");
            return error();
        }

        for (; vdbIt; ++vdbIt) {

            if (progress.wasInterrupted()) {
                throw std::runtime_error("processing was interrupted");
            }

            GU_PrimVDB* vdbPrim = *vdbIt;
            openvdb::GridBase::Ptr inGrid = vdbPrim->getGridPtr();

            if (!inGrid->isType<openvdb::tools::PointDataGrid>()) continue;
            openvdb::tools::PointDataGrid::Ptr pointDataGrid = openvdb::gridPtrCast<openvdb::tools::PointDataGrid>(inGrid);

            openvdb::tools::PointDataTree::LeafCIter leafIter = pointDataGrid->tree().cbeginLeaf();
            if (!leafIter) continue;

            bool hasDelayedLeaves = false;

            for (; leafIter; ++leafIter)
            {
                if (leafIter->buffer().isOutOfCore()) {
                    hasDelayedLeaves = true;
                    break;
                }
            }

            if (!hasDelayedLeaves)  continue;

            // deep copy the VDB tree if it is not already unique
            vdbPrim->makeGridUnique();

            openvdb::tools::PointDataGrid::Ptr outputGrid = openvdb::gridPtrCast<openvdb::tools::PointDataGrid>(vdbPrim->getGridPtr());

            if (!outputGrid) {
                addError(SOP_MESSAGE, "Failed to duplicate VDB Points");
                return error();
            }

            leafIter = outputGrid->tree().cbeginLeaf();

            for (; leafIter; ++leafIter) {
                // load out of core leaf nodes
                if (leafIter->buffer().isOutOfCore())    leafIter->buffer().data();
            }
        }

        return error();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}


////////////////////////////////////////

// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
