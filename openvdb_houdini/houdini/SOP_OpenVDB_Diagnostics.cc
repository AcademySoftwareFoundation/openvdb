///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
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
///
/// @todo Add more types of tests for volumes of type "Other"

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
        CheckFinite(bool _makeMask = false) : str(), makeMask(_makeMask) {}
        template<typename GridT>
        void operator() (const GridT& grid) {
            openvdb::tools::Diagnose<GridT> d(grid);
            openvdb::tools::CheckFinite<GridT,typename GridT::ValueAllCIter> c;
            str = d.check(c, makeMask, /*voxel*/true, /*tiles*/true, /*background*/true);
            if (makeMask) mask = d.mask();
        }
        std::string str;
        const bool makeMask;
        openvdb::BoolGrid::Ptr mask;
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

    // Tabs
    std::vector<PRM_Default> tab_parms;
    tab_parms.push_back(PRM_Default(11,"Level Sets"));
    tab_parms.push_back(PRM_Default(5, "FOG Volumes"));
    tab_parms.push_back(PRM_Default(9, "Other Volumes"));
    parms.add(hutil::ParmFactory(PRM_SWITCHER,
                                 PRMswitcherName.getToken(),
                                 PRMswitcherName.getLabel())
              .setVectorSize(3)
              .setDefault(tab_parms));

    const char* items1[] = {"off","disabled", "on", "enabled", NULL};
    const char* items2[] = {"off","disabled",
                            "onoff", "enabled w/o mask",
                            "onon",  "enabled with mask",NULL};
    
    ///////////// Level Set Options

    parms.add(hutil::ParmFactory(PRM_ORD, "CheckLS", "Check level sets")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items1)
              .setHelpText("Master switch to perform diagnostics of level sets.")
              .setDefault(PRMoneDefaults));   
    
    parms.add(hutil::ParmFactory(PRM_ORD, "ScaleLS", "Uniform voxels")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items1)
              .setHelpText("Verify that the voxels are uniform.")
              .setDefault(PRMoneDefaults));
    
    parms.add(hutil::ParmFactory(PRM_ORD, "BackgLS", "Background value")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items1)
              .setTypeExtended(PRM_TYPE_JOIN_PAIR)
              .setHelpText("Verify that the background value is bounded by "
                           "the minimum narrow-band width.")
              .setDefault(PRMoneDefaults));
    parms.add(hutil::ParmFactory(PRM_FLT_J, "WidthLS", "Width")
              .setDefault(3.0)
              .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 5.0)
              .setHelpText("Minimum allowed half-width of the narrow band in voxel units."));
    
    parms.add(hutil::ParmFactory(PRM_ORD, "TilesLS", "No active tiles")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items1)
              .setHelpText("Verify that all the tiles are inactive")
              .setDefault(PRMoneDefaults));
    
    parms.add(hutil::ParmFactory(PRM_ORD, "NaNLS", "No NaN values")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items2)
              .setHelpText("Verify that none of the values are NaN or infinite."
                           "Optionally generate a bool grid masking all "
                           "the values faling the test.")
              .setDefault(PRMoneDefaults));
    
    parms.add(hutil::ParmFactory(PRM_ORD, "ActiveLS", "Active values")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items2)
              .setHelpText("Verify that the active values are bounded by +/- the "
                           "background value. Optionally generate a bool grid masking "
                           "all the values faling the test.")
              .setDefault(PRMoneDefaults));
    
    parms.add(hutil::ParmFactory(PRM_ORD, "InactiveLS", "Inactive values")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items2)
              .setHelpText("Verify that the inactive values equal +/- the background value."
                           "Optionally generate a bool grid masking all "
                           "the values faling the test.")
              .setDefault(PRMoneDefaults));
    
    parms.add(hutil::ParmFactory(PRM_ORD, "GradLS", "Gradient norm")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items2)
              .setTypeExtended(PRM_TYPE_JOIN_PAIR)
              .setHelpText("Verify that norm of the gradient is bounded."
                           "Optionally generate a bool grid masking all "
                           "the values faling the test.")
              .setDefault(PRMzeroDefaults));
    parms.add(hutil::ParmFactory(PRM_FLT_J, "MinGrad", "Min")
              .setDefault(0.5)
              .setTypeExtended(PRM_TYPE_JOIN_PAIR)
              .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 2.0)
              .setHelpText("Minimum length of the gradient vector"));
    parms.add(hutil::ParmFactory(PRM_FLT_J, "MaxGrad", "Max")
              .setDefault(1.5)
              .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 2.0)
              .setHelpText("Maximum length of the gradient vector"));
    

    ///////////// FOG Volume Options

    parms.add(hutil::ParmFactory(PRM_ORD, "CheckFOG", "Check FOG volumes")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items1)
              .setHelpText("Master switch to perform diagnostics of FOG volumes.")
              .setDefault(PRMoneDefaults));
    
    parms.add(hutil::ParmFactory(PRM_ORD, "BackgFOG", "Background value")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items1)
              .setHelpText("Verify that the background value is zero.")
              .setDefault(PRMoneDefaults));
    
    parms.add(hutil::ParmFactory(PRM_ORD, "NaNFOG", "No NaN values")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items2)
              .setHelpText("Verify that none of the values are NaN or infinite."
                           "Optionally generate a bool grid masking all "
                           "the values faling the test.")
              .setDefault(PRMoneDefaults));
    
    parms.add(hutil::ParmFactory(PRM_ORD, "ActiveFOG", "Active values")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items2)
              .setHelpText("Verify that the active values are bounded by 0->1."
                           "Optionally generate a bool grid masking "
                           "all the values faling the test.")
              .setDefault(PRMoneDefaults));
    
    parms.add(hutil::ParmFactory(PRM_ORD, "InactiveFOG", "Inactive values")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items2)
              .setHelpText("Verify that the inactive values are zero."
                           "Optionally generate a bool grid masking "
                           "all the values faling the test.")
              .setDefault(PRMoneDefaults));
    
    ///////////// Other Volume Options
    
    parms.add(hutil::ParmFactory(PRM_ORD, "CheckOther", "Check other volumes")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items1)
              .setHelpText("Select the type of the voxel values.")
              .setDefault(PRMzeroDefaults));

    parms.add(hutil::ParmFactory(PRM_ORD, "NaNOther", "No NaN values")
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, items2)
              .setHelpText("Verify that none of the values are NaN or infinite."
                           "Optionally generate a bool grid masking all "
                           "the values faling the test.")
              .setDefault(PRMoneDefaults));
   
    ///////////////////////

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

    // Level Sets
    const bool checkLS = evalInt("CheckLS", 0, 0) == 1;
    changed |= enableParm("ScaleLS",   checkLS);
    changed |= enableParm("BackgLS",   checkLS);
    changed |= enableParm("WidthLS",   checkLS && evalInt("BackgLS",0,0)==1);
    changed |= enableParm("TilesLS",   checkLS);
    changed |= enableParm("NaNLS",     checkLS);
    changed |= enableParm("ActiveLS",  checkLS);
    changed |= enableParm("InactiveLS",checkLS);
    changed |= enableParm("GradLS",    checkLS);
    changed |= enableParm("MinGrad",   checkLS && evalInt("GradLS",0,0)>0);
    changed |= enableParm("MaxGrad",   checkLS && evalInt("GradLS",0,0)>0);

    // FOG Volumes
    const bool checkFOG = evalInt("CheckFOG", 0, 0) == 1;
    changed |= enableParm("BackgFOG",  checkFOG);
    changed |= enableParm("NaNFOG",    checkFOG);
    changed |= enableParm("ActiveFOG", checkFOG);
    changed |= enableParm("InactiveFOG",checkFOG);

    // Other Volumes
    changed |= enableParm("TypesOther",  evalInt("CheckOther", 0, 0) > 0);
    changed |= enableParm("NaNOther",    evalInt("CheckOther", 0, 0) > 0);
    
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
            
            if (gridClass == openvdb::GRID_LEVEL_SET && evalInt("CheckLS",0,0)>0) {
                if (vdbPrim->getStorageType() != UT_VDB_FLOAT) {
                    ss << "failed a level set test: Value type is not floating point\n";
                    ++nErr;
                } else {
                    
                    openvdb::FloatGrid& grid = UTvdbGridCast<openvdb::FloatGrid>(vdbPrim->getGrid());
                    openvdb::tools::CheckLevelSet<openvdb::FloatGrid> c(grid);
                    std::string str = evalInt("ScaleLS",0,0)==1 ? c.checkTransform() : "";
                    if (str.empty() && evalInt("BackgLS",0,0)==1) {
                        str = c.checkBackground(evalFloat("WidthLS",0,0));
                    }
                    if (str.empty() && evalInt("TilesLS",0,0)==1) {
                        str = c.checkTiles();
                    }
                    if (str.empty() && evalInt("NaNLS",0,0)>0) {
                        str = c.checkFinite(evalInt("NaNLS",0,0)==2);
                    }
                    if (str.empty() && evalInt("ActiveLS",0,0)>0) {
                        str = c.checkRange(evalInt("ActiveLS",0,0)==2);
                    }
                    if (str.empty() && evalInt("InactiveLS",0,0)>0) {
                        str = c.checkInactiveValues(evalInt("InactiveLS",0,0)==2);
                    }
                    if (str.empty() && evalInt("GradLS",0,0)>0) {
                        str = c.checkEikonal(evalInt("GradLS",0,0)==2,
                                             float(evalFloat("MinGrad",0,0)),
                                             float(evalFloat("MaxGrad",0,0)));
                    }
                    if (str.empty()) {
                        ss << "passed all level set tests\n";
                    } else {
                        ss << "failed a level set test: " << str;
                        ++nErr;
                    }
                    if (!c.mask()->empty()) {
                        std::ostringstream tmp;
                        tmp << it.getPrimitiveName("unnamed") << "_mask";
                        hvdb::createVdbPrimitive(*gdp, c.mask(), tmp.str().c_str());
                    }
                }
            } else if (gridClass == openvdb::GRID_FOG_VOLUME && evalInt("CheckFOG",0,0)>0) { 
                if (vdbPrim->getStorageType() != UT_VDB_FLOAT) {
                    ss << "failed a FOG volume test: Value type is not floating point\n";
                    ++nErr;
                } else {
                    openvdb::FloatGrid& grid = UTvdbGridCast<openvdb::FloatGrid>(vdbPrim->getGrid());
                    openvdb::tools::CheckFogVolume<openvdb::FloatGrid> c(grid);
                    std::string str = evalInt("BackgFOG",0,0)==1 ? c.checkBackground() : "";
                    if (str.empty() && evalInt("NaNFOG",0,0)>0) {
                        str = c.checkFinite(evalInt("NaNFOG",0,0)==2);
                    }
                    if (str.empty() && evalInt("ActiveFOG",0,0)>0) {
                        str = c.checkRange(evalInt("ActiveFOG",0,0)==2);
                    }
                    if (str.empty() && evalInt("InactiveFOG",0,0)>0) {
                        str = c.checkInactiveValues(evalInt("InactiveFOG",0,0)==2);
                    }
                    if (str.empty()) {
                        ss << "passed all FOG volume tests\n";
                    } else {
                        ss << "failed a FOG volume test: " << str;
                        ++nErr;
                    }
                    if (!c.mask()->empty()) {
                        std::ostringstream tmp;
                        tmp << it.getPrimitiveName("unnamed") << "_mask";
                        hvdb::createVdbPrimitive(*gdp, c.mask(), tmp.str().c_str());
                    }
                }
            } else if ((gridClass == openvdb::GRID_UNKNOWN ||
                        gridClass == openvdb::GRID_STAGGERED) && evalInt("CheckOther",0,0)>0) {
                std::string str;
                if (gridClass == openvdb::GRID_STAGGERED &&
                    (vdbPrim->getStorageType()!=UT_VDB_VEC3F ||
                     vdbPrim->getStorageType()!=UT_VDB_VEC3D) ) {
                    ss << "failed a test: staggered grid should contain vector values!\n";
                    str = ss.str();
                }
                openvdb::BoolGrid::Ptr mask;
                if (str.empty() && evalInt("NaNOther",0,0)>0) {
                    CheckFinite c(evalInt("NaNOther",0,0)==2);
                    GEOvdbProcessTypedGridTopology(*vdbPrim, c, /*makeUnique=*/false);
                    str = c.str;
                    mask = c.mask;
                    if (evalInt("NaNOther",0,0)==2 && !c.mask->empty()) {
                        std::ostringstream tmp;
                        tmp << it.getPrimitiveName("unnamed") << "_mask";
                        hvdb::createVdbPrimitive(*gdp, c.mask, tmp.str().c_str());
                    }
                }
                
                if (str.empty()) {
                    ss << "passed all tests\n";
                } else {
                    ss << "failed a test: " << str;
                    ++nErr;
                }
                if (mask && !mask->empty()) {
                    std::ostringstream tmp;
                    tmp << it.getPrimitiveName("unnamed") << "_mask";
                    hvdb::createVdbPrimitive(*gdp, mask, tmp.str().c_str());
                }
               
            } else {
                ss << "ignored!\n";
            }
        }//loop over vdb grids
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

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
