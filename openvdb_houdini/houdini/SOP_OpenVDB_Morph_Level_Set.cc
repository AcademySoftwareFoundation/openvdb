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
/// @file SOP_OpenVDB_Morph_Level_Set.cc
///
/// @author Ken Museth
///
/// @brief Level set morphing SOP

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/LevelSetMorph.h>
#include <boost/algorithm/string/join.hpp>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////

// Utilities

namespace {

struct MorphingParms {
    MorphingParms()
        : mLSGroup(NULL)
        , mAdvectSpatial(openvdb::math::UNKNOWN_BIAS)
        , mRenormSpatial(openvdb::math::UNKNOWN_BIAS)
        , mAdvectTemporal(openvdb::math::UNKNOWN_TIS)
        , mRenormTemporal(openvdb::math::UNKNOWN_TIS)
        , mNormCount(1)
        , mTimeStep(0.0)
        , mMinMask(0)
        , mMaxMask(1)
        , mInvertMask(false)
    {
    }

    const GA_PrimitiveGroup*  mLSGroup;
    openvdb::FloatGrid::ConstPtr mTargetGrid;
    openvdb::FloatGrid::ConstPtr mMaskGrid;
    openvdb::math::BiasedGradientScheme mAdvectSpatial, mRenormSpatial;
    openvdb::math::TemporalIntegrationScheme mAdvectTemporal, mRenormTemporal;
    int mNormCount;
    float mTimeStep;
    float mMinMask, mMaxMask;
    bool  mInvertMask;
};

class MorphOp
{
public:
    MorphOp(MorphingParms& parms, hvdb::Interrupter& boss)
        : mParms(&parms)
        , mBoss(&boss)
    {
    }

    void operator()(openvdb::FloatGrid& grid)
    {
        if (mBoss->wasInterrupted()) return;

        openvdb::tools::LevelSetMorphing<openvdb::FloatGrid, hvdb::Interrupter>
            morph(grid, *(mParms->mTargetGrid), mBoss);

        if (mParms->mMaskGrid) {
            morph.setAlphaMask(*(mParms->mMaskGrid));
            morph.setMaskRange(mParms->mMinMask, mParms->mMaxMask);
            morph.invertMask(mParms->mInvertMask);
        }
        morph.setSpatialScheme(mParms->mAdvectSpatial);
        morph.setTemporalScheme(mParms->mAdvectTemporal);
        morph.setTrackerSpatialScheme(mParms->mRenormSpatial);
        morph.setTrackerTemporalScheme(mParms->mRenormTemporal);
        morph.setNormCount(mParms->mNormCount);
        morph.advect(0, mParms->mTimeStep);
    }

private:
    MorphingParms*     mParms;
    hvdb::Interrupter* mBoss;
};

} // namespace


////////////////////////////////////////

// SOP Declaration

class SOP_OpenVDB_Morph_Level_Set: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Morph_Level_Set(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Morph_Level_Set() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i > 0); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

    OP_ERROR evalMorphingParms(OP_Context&, MorphingParms&);

    bool processGrids(MorphingParms&, hvdb::Interrupter&);
};

////////////////////////////////////////

// Build UI and register this operator

void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    using namespace openvdb::math;

    hutil::ParmList parms;

    // Level set grid
    parms.add(hutil::ParmFactory(PRM_STRING, "lsGroup", "Source Level Set")
              .setHelpText("Level set grid(s) to morph.")
              .setChoiceList(&hutil::PrimGroupMenuInput1));

    // Target grid
    parms.add(hutil::ParmFactory(PRM_STRING, "targetGroup", "Target Level Set")
              .setHelpText("Velocity grid")
              .setChoiceList(&hutil::PrimGroupMenuInput2));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "mask", "")
              .setDefault(PRMoneDefaults)
              .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
              .setHelpText("Enable / disable the mask."));

    // Alpha grid
    parms.add(hutil::ParmFactory(PRM_STRING, "maskGroup", "Alpha Mask")
         .setHelpText("Optional VDB used for alpha masking. Assumes values 0->1.")
        .setChoiceList(&hutil::PrimGroupMenuInput3));

    parms.add(hutil::ParmFactory(PRM_HEADING, "morphingHeading", "Morphing"));

    // Advect: timestep
    parms.add(hutil::ParmFactory(PRM_FLT, "timestep", "Time Step")
        .setDefault(1, "1.0/$FPS"));

    // Advect: spatial menu
    {
        std::vector<std::string> items;
        items.push_back(biasedGradientSchemeToString(FIRST_BIAS));
        items.push_back(biasedGradientSchemeToMenuName(FIRST_BIAS));

        items.push_back(biasedGradientSchemeToString(HJWENO5_BIAS));
        items.push_back(biasedGradientSchemeToMenuName(HJWENO5_BIAS));


        parms.add(hutil::ParmFactory(PRM_STRING, "advectSpatial", "Spatial Scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(1, ::strdup(biasedGradientSchemeToString(HJWENO5_BIAS).c_str()))
            .setHelpText("Set the spatial finite difference scheme."));
    }

    // Advect: temporal menu
    {
        std::vector<std::string> items;
        for (int i = 0; i < NUM_TEMPORAL_SCHEMES; ++i) {
            TemporalIntegrationScheme it = TemporalIntegrationScheme(i);
            items.push_back(temporalIntegrationSchemeToString(it)); // token
            items.push_back(temporalIntegrationSchemeToMenuName(it)); // label
        }

        parms.add(hutil::ParmFactory(PRM_STRING, "advectTemporal", "Temporal Scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(1, ::strdup(temporalIntegrationSchemeToString(TVD_RK2).c_str()))
            .setHelpText("Set the temporal integration scheme."));
    }

    parms.add(hutil::ParmFactory(PRM_HEADING, "renormHeading", "Renormalization"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "normSteps", "Steps")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setHelpText("The number of normalizations performed after each CFL iteration."));

    // Renorm: spatial menu
    {
        std::vector<std::string> items;
        items.push_back(biasedGradientSchemeToString(FIRST_BIAS));
        items.push_back(biasedGradientSchemeToMenuName(FIRST_BIAS));

        items.push_back(biasedGradientSchemeToString(HJWENO5_BIAS));
        items.push_back(biasedGradientSchemeToMenuName(HJWENO5_BIAS));

        parms.add(hutil::ParmFactory(PRM_STRING, "renormSpatial", "Spatial Scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(1, ::strdup(biasedGradientSchemeToString(HJWENO5_BIAS).c_str()))
            .setHelpText("Set the spatial finite difference scheme."));
    }

    // Renorm: temporal menu
    {
        std::vector<std::string> items;
        for (int i = 0; i < NUM_TEMPORAL_SCHEMES; ++i) {
            TemporalIntegrationScheme it = TemporalIntegrationScheme(i);
            items.push_back(temporalIntegrationSchemeToString(it)); // token
            items.push_back(temporalIntegrationSchemeToMenuName(it)); // label
        }

        parms.add(hutil::ParmFactory(PRM_STRING, "renormTemporal", "Temporal Scheme")
                  .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
                  .setDefault(1, ::strdup(temporalIntegrationSchemeToString(TVD_RK2).c_str()))
                  .setHelpText("Set the temporal integration scheme."));
    }


    parms.add(hutil::ParmFactory(PRM_HEADING, "maskHeading", "Alpha Mask"));

    //Invert mask.
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "invert", "Invert Alpha Mask")
              .setHelpText("Inverts the optional mask so alpha values 0->1 maps to 1->0"));

    // Min mask range
    parms.add(hutil::ParmFactory(PRM_FLT_J, "minMask", "Min Mask Cutoff")
              .setHelpText("Value below which the mask values map to zero")
              .setDefault(PRMzeroDefaults)
              .setRange(PRM_RANGE_UI, 0.0, PRM_RANGE_UI, 1.0));

    // Max mask range
    parms.add(hutil::ParmFactory(PRM_FLT_J, "maxMask", "Max Mask Cutoff")
              .setHelpText("Value above which the mask values map to one")
              .setDefault(PRMoneDefaults)
              .setRange(PRM_RANGE_UI, 0.0, PRM_RANGE_UI, 1.0));

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT, "beginTime", "Begin time"));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT, "endTime", "Time step"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Morph Level Set",
        SOP_OpenVDB_Morph_Level_Set::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("Source SDF VDBs to Morph")
        .addInput("Target SDF VDB")
        .addOptionalInput("Optional VDB Alpha Mask");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Morph_Level_Set::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Morph_Level_Set(net, name, op);
}


SOP_OpenVDB_Morph_Level_Set::SOP_OpenVDB_Morph_Level_Set(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////

// Enable/disable or show/hide parameters in the UI.

bool
SOP_OpenVDB_Morph_Level_Set::updateParmsFlags()
{
    bool changed = false;

    const bool hasMask = (this->nInputs() == 3);
    changed |= enableParm("mask", hasMask);
    const bool useMask = hasMask && bool(evalInt("mask", 0, 0));
    changed |= enableParm("invert",    useMask);
    changed |= enableParm("minMask",   useMask);
    changed |= enableParm("maxMask",   useMask);
    changed |= enableParm("maskGroup",useMask);

    return changed;
}



////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Morph_Level_Set::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        gdp->clearAndDestroy();
        duplicateSource(0, context);

        // Evaluate UI parameters
        MorphingParms parms;
        if (evalMorphingParms(context, parms) >= UT_ERROR_ABORT) return error();

        hvdb::Interrupter boss("Morphing level set");

        processGrids(parms, boss);

        if (boss.wasInterrupted()) addWarning(SOP_MESSAGE, "Process was interrupted");
        boss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Morph_Level_Set::evalMorphingParms(OP_Context& context, MorphingParms& parms)
{
    fpreal now = context.getTime();
    UT_String str;

    evalString(str, "lsGroup", 0, now);
    parms.mLSGroup = matchGroup(*gdp, str.toStdString());

    parms.mTimeStep = static_cast<float>(evalFloat("timestep", 0, now));

    evalString(str, "advectSpatial", 0, now);

    parms.mAdvectSpatial =
        openvdb::math::stringToBiasedGradientScheme(str.toStdString());

    if (parms.mAdvectSpatial == openvdb::math::UNKNOWN_BIAS) {
        addError(SOP_MESSAGE, "Morph: Unknown biased gradient");
        return UT_ERROR_ABORT;
    }

    evalString(str, "renormSpatial", 0, now);

    parms.mRenormSpatial =
        openvdb::math::stringToBiasedGradientScheme(str.toStdString());

    if (parms.mRenormSpatial == openvdb::math::UNKNOWN_BIAS) {
        addError(SOP_MESSAGE, "Renorm: Unknown biased gradient");
        return UT_ERROR_ABORT;
    }

    evalString(str, "advectTemporal", 0, now);
    parms.mAdvectTemporal =
        openvdb::math::stringToTemporalIntegrationScheme(str.toStdString());

    if (parms.mAdvectTemporal == openvdb::math::UNKNOWN_TIS) {
        addError(SOP_MESSAGE, "Morph: Unknown temporal integration");
        return UT_ERROR_ABORT;
    }

    evalString(str, "renormTemporal", 0, now);
    parms.mRenormTemporal =
        openvdb::math::stringToTemporalIntegrationScheme(str.toStdString());

    if (parms.mRenormTemporal == openvdb::math::UNKNOWN_TIS) {
        addError(SOP_MESSAGE, "Renorm: Unknown temporal integration");
        return UT_ERROR_ABORT;
    }

    parms.mNormCount = evalInt("normSteps", 0, now);

    const GU_Detail* targetGeo = inputGeo(1);

    if (!targetGeo) {
        addError(SOP_MESSAGE, "Missing target grid input");
        return UT_ERROR_ABORT;
    }

    evalString(str, "targetGroup", 0, now);
    const GA_PrimitiveGroup *targetGroup =
        matchGroup(const_cast<GU_Detail&>(*targetGeo), str.toStdString());

    hvdb::VdbPrimCIterator it(targetGeo, targetGroup);
    if (it) {
        if (it->getStorageType() != UT_VDB_FLOAT) {
            addError(SOP_MESSAGE, "Unrecognized target grid type.");
            return UT_ERROR_ABORT;
        }
        parms.mTargetGrid = hvdb::Grid::constGrid<openvdb::FloatGrid>(it->getConstGridPtr());
    }

    if (!parms.mTargetGrid) {
        addError(SOP_MESSAGE, "Missing target grid");
        return UT_ERROR_ABORT;
    }

    const GU_Detail* maskGeo = evalInt("mask", 0, now) ? inputGeo(2) : NULL;

    if (maskGeo) {
        evalString(str, "maskGroup", 0, now);
        const GA_PrimitiveGroup *maskGroup =
            matchGroup(const_cast<GU_Detail&>(*maskGeo), str.toStdString());

        hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);
        if (maskIt) {
            if (maskIt->getStorageType() != UT_VDB_FLOAT) {
                addError(SOP_MESSAGE, "Unrecognized alpha mask grid type.");
                return UT_ERROR_ABORT;
            }
            parms.mMaskGrid = hvdb::Grid::constGrid<openvdb::FloatGrid>(maskIt->getConstGridPtr());
        }

        if (!parms.mMaskGrid) {
            addError(SOP_MESSAGE, "Missing alpha mask grid");
            return UT_ERROR_ABORT;
        }
    }

    parms.mMinMask      = static_cast<float>(evalFloat("minMask", 0, now));
    parms.mMaxMask      = static_cast<float>(evalFloat("maxMask", 0, now));
    parms.mInvertMask   = evalInt("invert", 0, now);

    return error();
}


////////////////////////////////////////


bool
SOP_OpenVDB_Morph_Level_Set::processGrids(MorphingParms& parms, hvdb::Interrupter& boss)
{
    MorphOp op(parms, boss);

    std::vector<std::string> skippedGrids, nonLevelSetGrids, narrowBands;

    for (hvdb::VdbPrimIterator it(gdp, parms.mLSGroup); it; ++it) {

        if (boss.wasInterrupted()) break;

        GU_PrimVDB* vdbPrim = *it;

        const openvdb::GridClass gridClass = vdbPrim->getGrid().getGridClass();
        if (gridClass != openvdb::GRID_LEVEL_SET) {
            nonLevelSetGrids.push_back(it.getPrimitiveNameOrIndex().toStdString());
            continue;
        }

        if (vdbPrim->getStorageType() == UT_VDB_FLOAT) {
            vdbPrim->makeGridUnique();
            openvdb::FloatGrid& grid = UTvdbGridCast<openvdb::FloatGrid>(vdbPrim->getGrid());
            if ( grid.background() < float(openvdb::LEVEL_SET_HALF_WIDTH * grid.voxelSize()[0]) ) {
                narrowBands.push_back(it.getPrimitiveNameOrIndex().toStdString());
            }
            op(grid);
        } else {
            skippedGrids.push_back(it.getPrimitiveNameOrIndex().toStdString());
        }
    }

    if (!skippedGrids.empty()) {
        std::string s = "The following non-floating-point grids were skipped: "
            + boost::algorithm::join(skippedGrids, ", ");
        addWarning(SOP_MESSAGE, s.c_str());
    }

    if (!nonLevelSetGrids.empty()) {
        std::string s = "The following non-level-set grids were skipped: "
            + boost::algorithm::join(nonLevelSetGrids, ", ");
        addWarning(SOP_MESSAGE, s.c_str());
    }

    if (!narrowBands.empty()) {
        std::string s = "The following grids have a narrow band width that is"
            " less than 3 voxel units: " + boost::algorithm::join(narrowBands, ", ");
        addWarning(SOP_MESSAGE, s.c_str());
    }

    return true;
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
