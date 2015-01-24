///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Diagnostics.cc
///
/// @author Ken Museth
///
/// @brief Perform diagnostics on VDB volumes to detect potential issues.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>

#include <openvdb/tools/Diagnostics.h>
#include <openvdb/tools/Statistics.h>
#include <openvdb/tools/LevelSetUtil.h>

#include <UT/UT_Interrupt.h>
#if (UT_VERSION_INT >= 0x0c050157) // 12.5.343 or later
#include <GEO/GEO_PrimVDB.h> // for GEOvdbProcessTypedGridScalar(), etc.
#endif

namespace cvdb = openvdb;
namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Diagnostics: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Diagnostics(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Diagnostics() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i == 1); }

    static const char* sOpName[];

protected:

    struct CheckFinite {
        CheckFinite() : str() {}
        template<typename GridT>
        void operator() (const GridT& grid) {
            openvdb::tools::Diagnose<GridT> d(grid);
            openvdb::tools::CheckFinite<GridT,typename GridT::ValueAllCIter> c;
            str = d.check(c, false, /*voxel*/true, /*tiles*/true, /*background*/true);
        }
        std::string str;
    };
    
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
};

////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Group pattern
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to be processed.")
        .setChoiceList(&hutil::PrimGroupMenu));
   
    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Diagnostics", SOP_OpenVDB_Diagnostics::factory, parms, *table)
        .addInput("VDBs to diagnose");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Diagnostics::factory(OP_Network* net, const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Diagnostics(net, name, op);
}


SOP_OpenVDB_Diagnostics::SOP_OpenVDB_Diagnostics(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_Diagnostics::updateParmsFlags()
{
    bool changed = false;
    
    return changed;
}


OP_ERROR
SOP_OpenVDB_Diagnostics::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal time = context.getTime();

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSource(0, context);

        // Get the group of grids to be transformed.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        hvdb::Interrupter boss("Performing diagnostics");

        int nErr = 0;
        std::ostringstream ss;
        ss << "VDB Diagnostics:\n";
        // For each VDB primitive (with a non-null grid pointer) in the given group...
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (boss.wasInterrupted()) throw std::runtime_error("was interrupted");

            GU_PrimVDB* vdbPrim = *it;
            ss << it.getIndex() <<" (" << it.getPrimitiveName("unnamed") << ") ";
            
            const openvdb::GridClass gridClass = vdbPrim->getGrid().getGridClass();
            if (gridClass == openvdb::GRID_LEVEL_SET) {//level set
                if (vdbPrim->getStorageType() != UT_VDB_FLOAT) {
                    ss << "failed a level set test: Value type is not floating point\n";
                    ++nErr;
                } else {
                    openvdb::FloatGrid& grid = UTvdbGridCast<openvdb::FloatGrid>(vdbPrim->getGrid());
                    const std::string str = openvdb::tools::checkLevelSet(grid);
                    if (str.empty()) {
                        ss << "passed all level set tests\n";
                    } else {
                        ss << "failed a level set test: " << str;
                        ++nErr;
                    }
                }
            } else if (gridClass == openvdb::GRID_FOG_VOLUME) {//fog volume 
                if (vdbPrim->getStorageType() != UT_VDB_FLOAT) {
                    ss << "failed a FOG volume test: Value type is not floating point\n";
                    ++nErr;
                } else {
                    openvdb::FloatGrid& grid = UTvdbGridCast<openvdb::FloatGrid>(vdbPrim->getGrid());
                    const std::string str = openvdb::tools::checkFogVolume(grid);
                    if (str.empty()) {
                        ss << "passed all FOG volume tests\n";
                    } else {
                        ss << "failed a FOG volume test: " << str;
                        ++nErr;
                    }
                }
            } else {//unknown grid class
                CheckFinite c;
                GEOvdbProcessTypedGridTopology(*vdbPrim, c, /*makeUnique=*/false);
                if (c.str.empty()) {
                    ss << "passed all tests\n";
                } else {
                    ss << "failed a test: " << c.str;
                    ++nErr;
                }
            }
        }
        addMessage(SOP_MESSAGE, ss.str().c_str());
        ss.str("");
        if (nErr>0) {
            ss << nErr << " VDB grid" << (nErr==1?" ":"s ") << "failed the diagnostics test!";
            addWarning(SOP_MESSAGE, ss.str().c_str());
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
