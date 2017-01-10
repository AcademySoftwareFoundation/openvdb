///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_LOD.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Generate one or more levels of a volume mipmap.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/Utils.h>

#include <openvdb/tools/MultiResGrid.h>

#include <boost/algorithm/string/join.hpp>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_LOD: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_LOD(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_LOD() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned input) const { return (input > 0); }

protected:
    virtual bool updateParmsFlags();
    virtual OP_ERROR cookMySop(OP_Context&);
};


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to be processed.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    {
        const char* items[] = {
            "single", "Single Level",
            "range",  "Level Range",
            "mipmaps","LOD Pyramid",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "lod", "LOD Mode")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }



    parms.add(hutil::ParmFactory(PRM_FLT_J, "level", "Level")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 10.0)
        .setHelpText("Specify which level to produce.\n"
            "Level 0 is the highest-resolution level."));

    {
        std::vector<fpreal> defaultRange;
        defaultRange.push_back(fpreal(0.0)); // start
        defaultRange.push_back(fpreal(2.0)); // end
        defaultRange.push_back(fpreal(1.0)); // step

        parms.add(hutil::ParmFactory(PRM_FLT_J, "range", "Range")
            .setDefault(defaultRange)
            .setVectorSize(3)
            .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 10.0)
            .setHelpText("Specify the inclusive range [start, end, step]"));
    }

    parms.add(hutil::ParmFactory(PRM_INT_J, "count", "Count")
        .setDefault(PRMtwoDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 2, PRM_RANGE_UI, 10)
        .setHelpText("Number of levels\n"
            "Each level is half the resolution of the previous level."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "reuse", "Preserve Grid Names")
        .setHelpText("Reuse the name of the input VDB grid.\n"
            "Only available when a single Level is generated.")
        .setDefault(PRMzeroDefaults));

    hvdb::OpenVDBOpFactory("OpenVDB LOD", SOP_OpenVDB_LOD::factory, parms, *table)
        .addInput("VDBs");
}


bool
SOP_OpenVDB_LOD::updateParmsFlags()
{
    bool changed = false;

    int lodMode = evalInt("lod", 0, 0);

    changed |= enableParm("level", lodMode == 0);
    changed |= enableParm("reuse", lodMode == 0);
    changed |= enableParm("range", lodMode == 1);
    changed |= enableParm("count", lodMode == 2);

    return changed;
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_LOD::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_LOD(net, name, op);
}


SOP_OpenVDB_LOD::SOP_OpenVDB_LOD(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


namespace {

template<openvdb::Index Order>
struct MultiResGridFractionalOp
{
    MultiResGridFractionalOp(float f) : level(f) {}
    template<typename GridType>
    void operator()(const GridType& grid)
    {
        if ( level <= 0.0f ) {
            outputGrid = typename GridType::Ptr( new GridType(grid) );
        } else {
            const size_t levels = openvdb::math::Ceil(level) + 1;
            typedef typename GridType::TreeType TreeT;
            openvdb::tools::MultiResGrid<TreeT> mrg( levels, grid );
            outputGrid = mrg.template createGrid<Order>( level );
        }
    }
    const float level;
    hvdb::GridPtr outputGrid;
};

template<openvdb::Index Order>
struct MultiResGridRangeOp
{
    MultiResGridRangeOp(float start_, float end_, float step_, hvdb::Interrupter& boss_)
        : start(start_), end(end_), step(step_), outputGrids(), boss(&boss_)
    {}

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        if ( end > 0.0f ) {
            const size_t levels = openvdb::math::Ceil(end) + 1;
            typedef typename GridType::TreeType TreeT;
            openvdb::tools::MultiResGrid<TreeT> mrg( levels, grid );

            // inclusive range
            for (float level = start; !(level > end); level += step) {

                if (boss->wasInterrupted()) break;

                outputGrids.push_back( mrg.template createGrid<Order>( level ) );
            }
        }
    }
    const float start, end, step;
    std::vector<hvdb::GridPtr> outputGrids;
    hvdb::Interrupter * const boss;
};

struct MultiResGridIntegerOp
{
    MultiResGridIntegerOp(size_t n) : levels(n) {}
    template<typename GridType>
    void operator()(const GridType& grid)
    {
        typedef typename GridType::TreeType TreeT;
        openvdb::tools::MultiResGrid<TreeT> mrg( levels, grid );
        outputGrids = mrg.grids();
    }
    const size_t levels;
    openvdb::GridPtrVecPtr outputGrids;
};


inline bool
isValidRange(float start, float end, float step)
{
    if (start < 0.0f || !(step > 0.0f) || end < 0.0f) {
        return false;
    }

    return !(start > end);
}

}//unnamed namespace


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_LOD::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSource(0, context);

        // Get the group of grids to process.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        std::vector<std::string> skipped;

        hvdb::Interrupter boss("LOD");

        const int lodMode = evalInt("lod", 0, 0);
        if (lodMode == 0) {

            const bool reuseName = evalInt("reuse", 0, 0) > 0;
            MultiResGridFractionalOp<1> op( float(evalFloat("level", 0, time)) );
            for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {

                if (boss.wasInterrupted()) return error();

                if (!it->getGrid().transform().isLinear()) {
                    skipped.push_back(it->getGrid().getName());
                    continue;
                }

                GEOvdbProcessTypedGridTopology(**it, op);

                if (boss.wasInterrupted()) return error();

                if (reuseName) op.outputGrid->setName( it->getGrid().getName() );

                hvdb::createVdbPrimitive(*gdp, op.outputGrid);

                gdp->destroyPrimitiveOffset(it->getMapOffset(), /*and_points=*/true);
            }

        } else if (lodMode == 1) {

            const float start = float(evalFloat("range", 0, time));
            const float end = float(evalFloat("range", 1, time));
            const float step = float(evalFloat("range", 2, time));

            if (!isValidRange(start, end, step)) {
                addError(SOP_MESSAGE, "Invalid range, make sure that "
                    "start <= end and the step size is a positive number.");
                return error();
            }

            MultiResGridRangeOp<1> op( start, end, step, boss );
            for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {

                if (boss.wasInterrupted()) return error();

                if (!it->getGrid().transform().isLinear()) {
                    skipped.push_back(it->getGrid().getName());
                    continue;
                }

                GEOvdbProcessTypedGridTopology(**it, op);

                if (boss.wasInterrupted()) return error();

                for (size_t i=0; i< op.outputGrids.size(); ++i) {
                    hvdb::createVdbPrimitive(*gdp, op.outputGrids[i]);
                }

                gdp->destroyPrimitiveOffset(it->getMapOffset(), /*and_points=*/true);
            }
        } else if (lodMode == 2) {

            MultiResGridIntegerOp op( evalInt("count", 0, time) );
            for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {

                if (boss.wasInterrupted()) return error();

                if (!it->getGrid().transform().isLinear()) {
                    skipped.push_back(it->getGrid().getName());
                    continue;
                }

                GEOvdbProcessTypedGridTopology(**it, op);

                if (boss.wasInterrupted()) return error();

                for (size_t i=0; i< op.outputGrids->size(); ++i) {
                    hvdb::createVdbPrimitive(*gdp, op.outputGrids->at(i));
                }

                gdp->destroyPrimitiveOffset(it->getMapOffset(), /*and_points=*/true);
            }

        } else {
            addError(SOP_MESSAGE, "Invalid LOD option.");
        }

        if (!skipped.empty()) {
            addWarning(SOP_MESSAGE, ("Unable to process grid(s): " +
                boost::algorithm::join(skipped, ", ")).c_str());
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
