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
/// @file SOP_OpenVDB_Filter_Level_Set.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Performs various types of level set deformations with
/// interface tracking. These unrestricted deformations include
/// surface smoothing (e.g., Laplacian flow), filtering (e.g., mean
/// value) and morphological operations (e.g., morphological opening).
/// All these operations can optionally be masked with another grid that
/// acts as an alpha-mask.
///
/// @note Works with level set grids of floating point type (float/double).

#include <houdini_utils/OP_NodeChain.h> // for getNodeChain(), OP_EvalScope
#include <houdini_utils/ParmFactory.h>

#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/LevelSetFilter.h>

#include <OP/OP_AutoLockInputs.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_Version.h>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

#undef DWA_DEBUG_MODE
//#define DWA_DEBUG_MODE


////////////////////////////////////////

// Utilities

namespace {

// Add new items to the *end* of this list, and update NUM_OPERATOR_TYPES.
enum OperatorType {
    OP_TYPE_RENORM = 0,
    OP_TYPE_RESHAPE,
    OP_TYPE_SMOOTH,
    OP_TYPE_RESIZE
};

enum { NUM_OPERATOR_TYPES = OP_TYPE_RESIZE + 1 };


// Add new items to the *end* of this list, and update NUM_FILTER_TYPES.
enum FilterType {
    FILTER_TYPE_NONE = -1,
    FILTER_TYPE_RENORMALIZE = 0,
    FILTER_TYPE_MEAN_VALUE,
    FILTER_TYPE_MEDIAN_VALUE,
    FILTER_TYPE_MEAN_CURVATURE,
    FILTER_TYPE_LAPLACIAN_FLOW,
    FILTER_TYPE_DILATE,
    FILTER_TYPE_ERODE,
    FILTER_TYPE_OPEN,
    FILTER_TYPE_CLOSE,
    FILTER_TYPE_TRACK,
    FILTER_TYPE_GAUSSIAN,
    FILTER_TYPE_RESIZE
};

enum { NUM_FILTER_TYPES = FILTER_TYPE_RESIZE + 1 };


std::string
filterTypeToString(FilterType filter)
{
    std::string ret;
    switch (filter) {
        case FILTER_TYPE_NONE:           ret = "none";                  break;
        case FILTER_TYPE_RENORMALIZE:    ret = "renormalize";           break;
        case FILTER_TYPE_RESIZE:         ret = "resize narrow band";    break;
        case FILTER_TYPE_GAUSSIAN:       ret = "gaussian";              break;
        case FILTER_TYPE_DILATE:         ret = "dilate";                break;
        case FILTER_TYPE_ERODE:          ret = "erode";                 break;
        case FILTER_TYPE_OPEN:           ret = "open";                  break;
        case FILTER_TYPE_CLOSE:          ret = "close";                 break;
        case FILTER_TYPE_TRACK:          ret = "track";                 break;
#ifndef SESI_OPENVDB
        case FILTER_TYPE_MEAN_VALUE:     ret = "mean value";            break;
        case FILTER_TYPE_MEDIAN_VALUE:   ret = "median value";          break;
        case FILTER_TYPE_MEAN_CURVATURE: ret = "mean curvature";        break;
        case FILTER_TYPE_LAPLACIAN_FLOW: ret = "laplacian flow";        break;
#else
        case FILTER_TYPE_MEAN_VALUE:     ret = "meanvalue";             break;
        case FILTER_TYPE_MEDIAN_VALUE:   ret = "medianvalue";           break;
        case FILTER_TYPE_MEAN_CURVATURE: ret = "meancurvature";         break;
        case FILTER_TYPE_LAPLACIAN_FLOW: ret = "laplacianflow";         break;
#endif
    }
    return ret;
}

std::string
filterTypeToMenuName(FilterType filter)
{
    std::string ret;
    switch (filter) {
        case FILTER_TYPE_NONE: ret           = "None";                  break;
        case FILTER_TYPE_RENORMALIZE: ret    = "Renormalize";           break;
        case FILTER_TYPE_RESIZE: ret         = "Resize Narrow Band";    break;
        case FILTER_TYPE_MEAN_VALUE: ret     = "Mean Value";            break;
        case FILTER_TYPE_GAUSSIAN: ret       = "Gaussian";              break;
        case FILTER_TYPE_MEDIAN_VALUE: ret   = "Median Value";          break;
        case FILTER_TYPE_MEAN_CURVATURE: ret = "Mean Curvature Flow";   break;
        case FILTER_TYPE_LAPLACIAN_FLOW: ret = "Laplacian Flow";        break;
        case FILTER_TYPE_DILATE: ret         = "Dilate";                break;
        case FILTER_TYPE_ERODE: ret          = "Erode";                 break;
        case FILTER_TYPE_OPEN: ret           = "Open";                  break;
        case FILTER_TYPE_CLOSE: ret          = "Close";                 break;
        case FILTER_TYPE_TRACK: ret          = "Track Narrow Band";     break;
    }
    return ret;
}


FilterType
stringToFilterType(const std::string& s)
{
    FilterType ret = FILTER_TYPE_NONE;

    std::string str = s;
    boost::trim(str);
    boost::to_lower(str);

    if (str == filterTypeToString(FILTER_TYPE_RENORMALIZE)) {
        ret = FILTER_TYPE_RENORMALIZE;
    } else if (str == filterTypeToString(FILTER_TYPE_RESIZE)) {
        ret = FILTER_TYPE_RESIZE;
    } else if (str == filterTypeToString(FILTER_TYPE_MEAN_VALUE)) {
        ret = FILTER_TYPE_MEAN_VALUE;
    } else if (str == filterTypeToString(FILTER_TYPE_GAUSSIAN)) {
        ret = FILTER_TYPE_GAUSSIAN;
    } else if (str == filterTypeToString(FILTER_TYPE_MEDIAN_VALUE)) {
        ret = FILTER_TYPE_MEDIAN_VALUE;
    } else if (str == filterTypeToString(FILTER_TYPE_MEAN_CURVATURE)) {
        ret = FILTER_TYPE_MEAN_CURVATURE;
    } else if (str == filterTypeToString(FILTER_TYPE_LAPLACIAN_FLOW)) {
        ret = FILTER_TYPE_LAPLACIAN_FLOW;
    } else if (str == filterTypeToString(FILTER_TYPE_DILATE)) {
        ret = FILTER_TYPE_DILATE;
    } else if (str == filterTypeToString(FILTER_TYPE_ERODE)) {
        ret = FILTER_TYPE_ERODE;
    } else if (str == filterTypeToString(FILTER_TYPE_OPEN)) {
        ret = FILTER_TYPE_OPEN;
    } else if (str == filterTypeToString(FILTER_TYPE_CLOSE)) {
        ret = FILTER_TYPE_CLOSE;
    } else if (str == filterTypeToString(FILTER_TYPE_TRACK)) {
        ret = FILTER_TYPE_TRACK;
    }

    return ret;
}


// Add new items to the *end* of this list, and update NUM_ACCURACY_TYPES.
enum Accuracy {
    ACCURACY_UPWIND_FIRST = 0,
    ACCURACY_UPWIND_SECOND,
    ACCURACY_UPWIND_THIRD,
    ACCURACY_WENO,
    ACCURACY_HJ_WENO
};

enum { NUM_ACCURACY_TYPES = ACCURACY_HJ_WENO + 1 };

std::string
accuracyToString(Accuracy ac)
{
    std::string ret;
    switch (ac) {
        case ACCURACY_UPWIND_FIRST: ret     = "upwind first";       break;
        case ACCURACY_UPWIND_SECOND: ret    = "upwind second";      break;
        case ACCURACY_UPWIND_THIRD: ret     = "upwind third";       break;
        case ACCURACY_WENO: ret             = "weno";               break;
        case ACCURACY_HJ_WENO: ret          = "hj weno";            break;
    }
    return ret;
}

std::string
accuracyToMenuName(Accuracy ac)
{
    std::string ret;
    switch (ac) {
        case ACCURACY_UPWIND_FIRST: ret     = "First-order upwinding";      break;
        case ACCURACY_UPWIND_SECOND: ret    = "Second-order upwinding";     break;
        case ACCURACY_UPWIND_THIRD: ret     = "Third-order upwinding";      break;
        case ACCURACY_WENO: ret             = "Fifth-order WENO";           break;
        case ACCURACY_HJ_WENO: ret          = "Fifth-order HJ-WENO";        break;
    }
    return ret;
}


Accuracy
stringToAccuracy(const std::string& s)
{
    Accuracy ret = ACCURACY_UPWIND_FIRST;

    std::string str = s;
    boost::trim(str);
    boost::to_lower(str);

    if (str == accuracyToString(ACCURACY_UPWIND_SECOND)) {
        ret = ACCURACY_UPWIND_SECOND;
    } else if (str == accuracyToString(ACCURACY_UPWIND_THIRD)) {
        ret = ACCURACY_UPWIND_THIRD;
    } else if (str == accuracyToString(ACCURACY_WENO)) {
        ret = ACCURACY_WENO;
    } else if (str == accuracyToString(ACCURACY_HJ_WENO)) {
        ret = ACCURACY_HJ_WENO;
    }

    return ret;
}


void
buildFilterMenu(std::vector<std::string>& items, OperatorType op)
{
    items.clear();

    if (OP_TYPE_SMOOTH == op) {

        items.push_back(filterTypeToString(FILTER_TYPE_MEAN_VALUE));
        items.push_back(filterTypeToMenuName(FILTER_TYPE_MEAN_VALUE));

        items.push_back(filterTypeToString(FILTER_TYPE_GAUSSIAN));
        items.push_back(filterTypeToMenuName(FILTER_TYPE_GAUSSIAN));

        items.push_back(filterTypeToString(FILTER_TYPE_MEDIAN_VALUE));
        items.push_back(filterTypeToMenuName(FILTER_TYPE_MEDIAN_VALUE));

        items.push_back(filterTypeToString(FILTER_TYPE_MEAN_CURVATURE));
        items.push_back(filterTypeToMenuName(FILTER_TYPE_MEAN_CURVATURE));

        items.push_back(filterTypeToString(FILTER_TYPE_LAPLACIAN_FLOW));
        items.push_back(filterTypeToMenuName(FILTER_TYPE_LAPLACIAN_FLOW));

    } else if (OP_TYPE_RESHAPE == op) {

        items.push_back(filterTypeToString(FILTER_TYPE_DILATE));
        items.push_back(filterTypeToMenuName(FILTER_TYPE_DILATE));

        items.push_back(filterTypeToString(FILTER_TYPE_ERODE));
        items.push_back(filterTypeToMenuName(FILTER_TYPE_ERODE));

        items.push_back(filterTypeToString(FILTER_TYPE_OPEN));
        items.push_back(filterTypeToMenuName(FILTER_TYPE_OPEN));

        items.push_back(filterTypeToString(FILTER_TYPE_CLOSE));
        items.push_back(filterTypeToMenuName(FILTER_TYPE_CLOSE));

#ifdef DWA_DEBUG_MODE
        items.push_back(filterTypeToString(FILTER_TYPE_TRACK));
        items.push_back(filterTypeToMenuName(FILTER_TYPE_TRACK));
#endif

    }
}

struct FilterParms {
    FilterParms()
        : mGroup()
        , mMaskName()
        , mSecondInputConnected(false)
        , mFilterType(FILTER_TYPE_NONE)
        , mIterations(0)
        , mHalfWidth(3)
        , mStencilWidth(0)
        , mVoxelOffset(0.0)
        , mWorldUnits(false)
        , mMinMask(0)
        , mMaxMask(1)
        , mInvertMask(false)
        , mAccuracy(ACCURACY_UPWIND_FIRST)
        , mMaskInputNode(NULL)
    {
    }

    std::string mGroup, mMaskName;
    bool mSecondInputConnected;
    FilterType mFilterType;
    int mIterations, mHalfWidth, mStencilWidth;
    float mVoxelOffset;
    bool  mWorldUnits;
    float mMinMask, mMaxMask;
    bool  mInvertMask;
    Accuracy mAccuracy;
    OP_Node* mMaskInputNode;
};

} // namespace


////////////////////////////////////////

// SOP Declaration

class SOP_OpenVDB_Filter_Level_Set: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Filter_Level_Set(OP_Network*, const char* name, OP_Operator*, OperatorType);
    virtual ~SOP_OpenVDB_Filter_Level_Set() {}

    static OP_Node* factoryRenormalize(OP_Network*, const char* name, OP_Operator*);
    static OP_Node* factorySmooth(OP_Network*, const char* name, OP_Operator*);
    static OP_Node* factoryReshape(OP_Network*, const char* name, OP_Operator*);
    static OP_Node* factoryNarrowBand(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned input) const { return (input == 1); }

    int convertUnits();

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

private:
    typedef hvdb::Interrupter BossT;

    const OperatorType mOpType;
    float mVoxelSize;

    OP_ERROR evalFilterParms(OP_Context&, FilterParms&);


    template<typename GridT>
    bool applyFilters(GU_PrimVDB*, std::vector<FilterParms>&, BossT&,
        OP_Context&, GU_Detail&, bool verbose);

    template<typename FilterT>
    void filterGrid(OP_Context&, FilterT&, const FilterParms&, BossT&, bool verbose);

    template<typename FilterT>
    void offset(const FilterParms&, FilterT&, const float offset, bool verbose,
        const typename FilterT::MaskType* mask = NULL);

    template<typename FilterT>
    void mean(const FilterParms&, FilterT&, BossT&, bool verbose,
        const typename FilterT::MaskType* mask = NULL);

    template<typename FilterT>
    void gaussian(const FilterParms&, FilterT&, BossT&, bool verbose,
        const typename FilterT::MaskType* mask = NULL);

    template<typename FilterT>
    void median(const FilterParms&, FilterT&, BossT&, bool verbose,
        const typename FilterT::MaskType* mask = NULL);

    template<typename FilterT>
    void meanCurvature(const FilterParms&, FilterT&, BossT&, bool verbose,
        const typename FilterT::MaskType* mask = NULL);

    template<typename FilterT>
    void laplacian(const FilterParms&, FilterT&, BossT&, bool verbose,
        const typename FilterT::MaskType* mask = NULL);

    template<typename FilterT>
    void renormalize(const FilterParms&, FilterT&, BossT&, bool verbose = false);

    template<typename FilterT>
    void resizeNarrowBand(const FilterParms&, FilterT&, BossT&, bool verbose = false);

    template<typename FilterT>
    void track(const FilterParms&, FilterT&, BossT&, bool verbose);
};//SOP_OpenVDB_Filter_Level_Set

////////////////////////////////////////

namespace
{

// Callback to convert from voxel to world space units
int
convertUnitsCB(void* data, int /*idx*/, float /*time*/, const PRM_Template*)
{
   SOP_OpenVDB_Filter_Level_Set* sop = static_cast<SOP_OpenVDB_Filter_Level_Set*>(data);
   if (sop == NULL) return 0;
   return sop->convertUnits();
}

} // unnamed namespace

////////////////////////////////////////

// Build UI

void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    for (int n = 0; n < NUM_OPERATOR_TYPES; ++n) {

        OperatorType op = OperatorType(n);

        hutil::ParmList parms;

        // Define a string-valued group name pattern parameter and add it to the list.
        parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
            .setHelpText("Specify a subset of the input VDB grids to be processed.")
            .setChoiceList(&hutil::PrimGroupMenuInput1));

        if (OP_TYPE_RENORM != op && OP_TYPE_RESIZE != op) { // Filter menu

            parms.add(hutil::ParmFactory(PRM_TOGGLE, "mask", "")
                .setDefault(PRMoneDefaults)
                .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
                .setHelpText("Enable / disable the mask."));

            parms.add(hutil::ParmFactory(PRM_STRING, "maskname", "Alpha Mask")
                .setHelpText("Optional VDB used for alpha masking. Assumes values 0->1.")
                .setChoiceList(&hutil::PrimGroupMenuInput2));

            std::vector<std::string> items;

            buildFilterMenu(items, op);

            parms.add(hutil::ParmFactory(PRM_STRING, "operation", "Operation")
                .setDefault(items[0])
                .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));

        }

        // Stencil width
        parms.add(hutil::ParmFactory(PRM_INT_J, "stencilWidth", "Filter Voxel Radius")
            .setDefault(PRMoneDefaults)
            .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 5));

        // steps
        parms.add(hutil::ParmFactory(PRM_INT_J, "iterations", "Iterations")
            .setDefault(PRMfourDefaults)
            .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10));

        // Narrow-Band half-width
        parms.add(hutil::ParmFactory(PRM_INT_J, "halfWidth", "Half-Width")
            .setHelpText("Desired half-width in voxel units of the narrow-band "
                         "level set (3 is normally best).")
            .setDefault(PRMthreeDefaults)
            .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10));

        // Toggle between world- and index-space units for offset
        parms.add(hutil::ParmFactory(PRM_TOGGLE, "worldSpaceUnits",
                  "Specify Offset in World (vs Voxel) Units")
                  .setCallbackFunc(&convertUnitsCB));

        // Offset
        parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelOffset", "Offset")
                  .setDefault(PRMoneDefaults)
                  .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 10.0));

        { // Renormalization accuracy

            std::vector<std::string> items;
            for (int i = 0; i < NUM_ACCURACY_TYPES; ++i) {
                Accuracy ac = Accuracy(i);

#ifndef DWA_DEBUG_MODE // Exclude some of the menu options
                if (ac == ACCURACY_UPWIND_THIRD || ac == ACCURACY_WENO) continue;
#endif

                items.push_back(accuracyToString(ac)); // token
                items.push_back(accuracyToMenuName(ac)); // label
            }

            parms.add(hutil::ParmFactory(PRM_STRING, "accuracy", "Renorm Accuracy")
                .setDefault(items[0])
                .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
        }

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

#ifndef SESI_OPENVDB
        // Verbosity toggle.
        parms.add(hutil::ParmFactory(PRM_TOGGLE, "verbose", "Verbose")
            .setHelpText("Prints the sequence of operations to the terminal."));
#endif

        // Obsolete parameters
        hutil::ParmList obsoleteParms;
        obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep1", "Sep"));
        obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep2", "Sep"));

        // Register operator
        if (OP_TYPE_RENORM == op) {

            hvdb::OpenVDBOpFactory("OpenVDB Renormalize Level Set",
                SOP_OpenVDB_Filter_Level_Set::factoryRenormalize, parms, *table)
                .setObsoleteParms(obsoleteParms)
                .addInput("Input with VDB grids to process");

        } else if (OP_TYPE_RESHAPE == op) {

            hvdb::OpenVDBOpFactory("OpenVDB Offset Level Set",
                SOP_OpenVDB_Filter_Level_Set::factoryReshape, parms, *table)
                .setObsoleteParms(obsoleteParms)
                .addInput("Input with VDBs to process")
                .addOptionalInput("Optional VDB Alpha Mask");

        } else if (OP_TYPE_SMOOTH == op) {

            hvdb::OpenVDBOpFactory("OpenVDB Smooth Level Set",
                SOP_OpenVDB_Filter_Level_Set::factorySmooth, parms, *table)
                .setObsoleteParms(obsoleteParms)
                .addInput("Input with VDBs to process")
                .addOptionalInput("Optional VDB Alpha Mask");

        } else if (OP_TYPE_RESIZE == op) {

            hvdb::OpenVDBOpFactory("OpenVDB Resize Narrow Band",
                SOP_OpenVDB_Filter_Level_Set::factoryNarrowBand, parms, *table)
                .setObsoleteParms(obsoleteParms)
                .addInput("Input with VDBs to process");
        }
    }
 }

////////////////////////////////////////

// Operator registration

OP_Node*
SOP_OpenVDB_Filter_Level_Set::factoryRenormalize(
    OP_Network* net, const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Filter_Level_Set(net, name, op, OP_TYPE_RENORM);
}

OP_Node*
SOP_OpenVDB_Filter_Level_Set::factoryReshape(
    OP_Network* net, const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Filter_Level_Set(net, name, op, OP_TYPE_RESHAPE);
}

OP_Node*
SOP_OpenVDB_Filter_Level_Set::factorySmooth(
    OP_Network* net, const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Filter_Level_Set(net, name, op, OP_TYPE_SMOOTH);
}

OP_Node*
SOP_OpenVDB_Filter_Level_Set::factoryNarrowBand(
    OP_Network* net, const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Filter_Level_Set(net, name, op, OP_TYPE_RESIZE);
}

SOP_OpenVDB_Filter_Level_Set::SOP_OpenVDB_Filter_Level_Set(
    OP_Network* net, const char* name, OP_Operator* op, OperatorType opType)
    : hvdb::SOP_NodeVDB(net, name, op)
    , mOpType(opType), mVoxelSize(1.0f)
{
}

////////////////////////////////////////

// Disable UI Parms.
bool
SOP_OpenVDB_Filter_Level_Set::updateParmsFlags()
{
    bool changed = false, stencil = false;
    const bool renorm = mOpType == OP_TYPE_RENORM;
    const bool smooth = mOpType == OP_TYPE_SMOOTH;
    const bool reshape = mOpType == OP_TYPE_RESHAPE;
    const bool resize = mOpType == OP_TYPE_RESIZE;

    if (renorm || resize) {
        changed |= setVisibleState("invert", false);
        changed |= setVisibleState("minMask",false);
        changed |= setVisibleState("maxMask",false);
    } else {
        UT_String str;
        evalString(str, "operation", 0, 0);
        FilterType operation = stringToFilterType(str.toStdString());
        stencil = operation == FILTER_TYPE_MEAN_VALUE ||
                  operation == FILTER_TYPE_GAUSSIAN   ||
                  operation == FILTER_TYPE_MEDIAN_VALUE;
        const bool hasMask = (this->nInputs() == 2);
        changed |= enableParm("mask", hasMask);
        const bool useMask = hasMask && bool(evalInt("mask", 0, 0));
        changed |= enableParm("invert",   useMask);
        changed |= enableParm("minMask",  useMask);
        changed |= enableParm("maxMask",  useMask);
        changed |= enableParm("maskname", useMask);
    }

    changed |= setVisibleState("halfWidth", resize);

    changed |= enableParm("iterations",  smooth || renorm);
    changed |= enableParm("stencilWidth", stencil);

    changed |= setVisibleState("stencilWidth", getEnableState("stencilWidth"));
    changed |= setVisibleState("iterations",   getEnableState("iterations"));

    changed |= setVisibleState("worldSpaceUnits", reshape);
    changed |= setVisibleState("voxelOffset",     reshape);

    return changed;
}

////////////////////////////////////////


int
SOP_OpenVDB_Filter_Level_Set::convertUnits()
{
    const bool toWS = static_cast<bool>(evalInt("worldSpaceUnits", 0, 0));
    const fpreal offset = evalFloat("voxelOffset", 0, 0) * (toWS ? mVoxelSize : 1.0f/mVoxelSize);
    setFloat("voxelOffset", 0, 0, offset);
    return 1;
}

////////////////////////////////////////

// Cook

OP_ERROR
SOP_OpenVDB_Filter_Level_Set::cookMySop(OP_Context& context)
{
    try {
        OP_AutoLockInputs lock;
        std::vector<FilterParms> filterParms;
        SOP_OpenVDB_Filter_Level_Set* startNode = this;

        {
            // Find adjacent, upstream nodes of the same type as this node.
            std::vector<SOP_OpenVDB_Filter_Level_Set*> nodes =
                hutil::getNodeChain(context, this);

            startNode = nodes[0];

            // Collect filter parameters starting from the topmost node.
            filterParms.resize(nodes.size());
            for (size_t n = 0, N = filterParms.size(); n < N; ++n) {
                if (nodes[n]->evalFilterParms(context, filterParms[n]) >= UT_ERROR_ABORT) {
                    return error();
                }
            }
        }
#if (UT_VERSION_INT >= 0x0e0000b0) // 14.0.176 or later
        lock.setNode(startNode);
        if (lock.lock(context) >= UT_ERROR_ABORT) return error();
#else
        if (lock.lock(*startNode, context) >= UT_ERROR_ABORT) return error();
#endif

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        if (startNode->duplicateSource(0, context, gdp) >= UT_ERROR_ABORT) return error();

        BossT boss("Processing level sets");

        const fpreal time = context.getTime();
#ifndef SESI_OPENVDB
        const bool verbose = bool(evalInt("verbose", 0, time));
#else
        const bool verbose = false;
#endif

        if (verbose) std::cout << "--- " << this->getName() << " ---\n";

        // Filter grids
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);

        const GA_PrimitiveGroup *group =
            matchGroup(const_cast<GU_Detail&>(*gdp), groupStr.toStdString());
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {

            // Check grid class
            const openvdb::GridClass gridClass = it->getGrid().getGridClass();
            if (gridClass != openvdb::GRID_LEVEL_SET) {
                std::string s = it.getPrimitiveNameOrIndex().toStdString();
                s = "VDB primitive " + s + " was skipped because it is not a level-set grid.";
                addWarning(SOP_MESSAGE, s.c_str());
                continue;
            }

            // Appply filters

            bool wasFiltered = applyFilters<openvdb::FloatGrid>(
                *it, filterParms, boss, context, *gdp, verbose);

            if (boss.wasInterrupted()) break;

            if (!wasFiltered) {
                wasFiltered = applyFilters<openvdb::DoubleGrid>(
                    *it, filterParms, boss, context, *gdp, verbose);
            }

            if (boss.wasInterrupted()) break;

            if (!wasFiltered) {
                std::string msg = "VDB primitive "
                    + it.getPrimitiveNameOrIndex().toStdString()
                    + " is not of floating point type.";
                addWarning(SOP_MESSAGE, msg.c_str());
                continue;
            }

            if (boss.wasInterrupted()) break;
        }

        if (boss.wasInterrupted()) addWarning(SOP_MESSAGE, "processing was interrupted");
        boss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Filter_Level_Set::evalFilterParms(OP_Context& context,
    FilterParms& parms)
{
    hutil::OP_EvalScope eval_scope(*this, context);
    fpreal now = context.getTime();

    parms.mIterations   = evalInt("iterations", 0, now);
    parms.mHalfWidth    = evalInt("halfWidth", 0, now);
    parms.mStencilWidth = evalInt("stencilWidth", 0, now);
    parms.mVoxelOffset  = static_cast<float>(evalFloat("voxelOffset", 0, now));
    parms.mMinMask      = static_cast<float>(evalFloat("minMask", 0, now));
    parms.mMaxMask      = static_cast<float>(evalFloat("maxMask", 0, now));
    parms.mInvertMask   = evalInt("invert", 0, now);
    parms.mWorldUnits   = evalInt("worldSpaceUnits", 0, now);

    UT_String str;

    if (OP_TYPE_RENORM == mOpType ) {
        parms.mFilterType = FILTER_TYPE_RENORMALIZE;
    } else if (OP_TYPE_RESIZE == mOpType) {
        parms.mFilterType = FILTER_TYPE_RESIZE;
    } else {
        evalString(str, "operation", 0, now);
        parms.mFilterType = stringToFilterType(str.toStdString());
    }

    evalString(str, "accuracy", 0, now);
    parms.mAccuracy = stringToAccuracy(str.toStdString());

    evalString(str, "group", 0, now);
    parms.mGroup = str.toStdString();

    if (OP_TYPE_SMOOTH == mOpType || OP_TYPE_RESHAPE == mOpType) {
        if (evalInt("mask", 0, now)) {
            parms.mMaskInputNode = getInput(1, /*mark_used*/true);

            evalString(str, "maskname", 0, now);
            parms.mMaskName = str.toStdString();
        }
    }

    return error();
}


////////////////////////////////////////

// Filter callers

template<typename GridT>
bool
SOP_OpenVDB_Filter_Level_Set::applyFilters(
    GU_PrimVDB* vdbPrim,
    std::vector<FilterParms>& filterParms,
    BossT& boss,
    OP_Context& context,
    GU_Detail&,
    bool verbose)
{
    typename GridT::Ptr grid = openvdb::deepCopyTypedGrid<GridT>(vdbPrim->getGrid());

    if (!grid) return false;

    typedef typename GridT::ValueType ValueT;
    typedef openvdb::FloatGrid MaskT;
    typedef openvdb::tools::LevelSetFilter<GridT, MaskT, BossT> FilterT;

    mVoxelSize = static_cast<float>(grid->voxelSize()[0]);
    FilterT filter(*grid, &boss);
    filter.setTemporalScheme(openvdb::math::TVD_RK1);

    if (grid->background() < ValueT(openvdb::LEVEL_SET_HALF_WIDTH * mVoxelSize)) {
        std::string msg = "VDB primitive '"
            + std::string(vdbPrim->getGridName())
            + "' has a narrow band width that is less than 3 voxel units. ";
        addWarning(SOP_MESSAGE, msg.c_str());
    }

    for (size_t n = 0, N = filterParms.size(); n < N; ++n) {

        const GA_PrimitiveGroup *group = matchGroup(*gdp, filterParms[n].mGroup);

        // Skip this node if it doesn't operate on this primitive
        if (group && !group->containsOffset(vdbPrim->getMapOffset())) continue;

        filterGrid(context, filter, filterParms[n], boss, verbose);

        if (boss.wasInterrupted()) break;
    }

    // Replace the original VDB primitive with a new primitive that contains
    // the output grid and has the same attributes and group membership.
    hvdb::replaceVdbPrimitive(*gdp, grid, *vdbPrim, true, vdbPrim->getGridName());

    return true;
}


template<typename FilterT>
void
SOP_OpenVDB_Filter_Level_Set::filterGrid(OP_Context& context, FilterT& filter,
    const FilterParms& parms, BossT& boss, bool verbose)
{
    // Alpha-masking
    typedef typename FilterT::GridType GridT;
    typedef typename FilterT::MaskType MaskT;
    typename MaskT::ConstPtr maskGrid;

    if (parms.mMaskInputNode) {

        // record second input
        if (getInput(1) != parms.mMaskInputNode) {
            addExtraInput(parms.mMaskInputNode, OP_INTEREST_DATA);
        }

        GU_DetailHandle maskHandle;
        maskHandle = static_cast<SOP_Node*>(parms.mMaskInputNode)->getCookedGeoHandle(context);

        GU_DetailHandleAutoReadLock maskScope(maskHandle);
        const GU_Detail *maskGeo = maskScope.getGdp();

        if (maskGeo) {
            const GA_PrimitiveGroup * maskGroup =
                parsePrimitiveGroups(parms.mMaskName.c_str(), const_cast<GU_Detail*>(maskGeo));

            if (!maskGroup && !parms.mMaskName.empty()) {
                addWarning(SOP_MESSAGE, "Mask not found.");
            } else {
                hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);
                if (maskIt) {
                    if (maskIt->getStorageType() == UT_VDB_FLOAT) {
                        maskGrid = openvdb::gridConstPtrCast<MaskT>(maskIt->getGridPtr());
                    } else {
                        addWarning(SOP_MESSAGE, "The mask grid has to be a FloatGrid.");
                    }
                } else {
                    addWarning(SOP_MESSAGE, "The mask input is empty.");
                }
            }
        }
        filter.setMaskRange(parms.mMinMask, parms.mMaxMask);
        filter.invertMask(parms.mInvertMask);
    }

    switch (parms.mAccuracy) {
      case ACCURACY_UPWIND_FIRST:  filter.setSpatialScheme(openvdb::math::FIRST_BIAS);   break;
      case ACCURACY_UPWIND_SECOND: filter.setSpatialScheme(openvdb::math::SECOND_BIAS);  break;
      case ACCURACY_UPWIND_THIRD:  filter.setSpatialScheme(openvdb::math::THIRD_BIAS);   break;
      case ACCURACY_WENO:          filter.setSpatialScheme(openvdb::math::WENO5_BIAS);   break;
      case ACCURACY_HJ_WENO:       filter.setSpatialScheme(openvdb::math::HJWENO5_BIAS); break;
    }

    typedef typename FilterT::ValueType ValueT;
    const float ds = (parms.mWorldUnits ? 1.f : mVoxelSize) * parms.mVoxelOffset;

    switch (parms.mFilterType) {

        case FILTER_TYPE_NONE:
            break;
        case FILTER_TYPE_RENORMALIZE:
            renormalize(parms, filter, boss, verbose);
            break;
        case FILTER_TYPE_RESIZE:
            resizeNarrowBand(parms, filter, boss, verbose);
            break;
        case FILTER_TYPE_MEAN_VALUE:
            mean(parms, filter, boss, verbose, maskGrid.get());
            break;
        case FILTER_TYPE_GAUSSIAN:
            gaussian(parms, filter, boss, verbose, maskGrid.get());
            break;
        case FILTER_TYPE_MEDIAN_VALUE:
            median(parms, filter, boss, verbose, maskGrid.get());
            break;
        case FILTER_TYPE_MEAN_CURVATURE:
            meanCurvature(parms, filter, boss, verbose, maskGrid.get());
            break;
        case FILTER_TYPE_LAPLACIAN_FLOW:
            laplacian(parms, filter, boss, verbose, maskGrid.get());
            break;
        case FILTER_TYPE_TRACK:
            track(parms, filter, boss, verbose);
            break;
        case FILTER_TYPE_DILATE:
            offset(parms, filter, -ds, verbose, maskGrid.get());
            break;
        case FILTER_TYPE_ERODE:
            offset(parms, filter,  ds, verbose, maskGrid.get());
            break;
        case FILTER_TYPE_OPEN:
            offset(parms, filter,  ds, verbose, maskGrid.get());
            offset(parms, filter, -ds, verbose, maskGrid.get());
            break;
        case FILTER_TYPE_CLOSE:
            offset(parms, filter, -ds, verbose, maskGrid.get());
            offset(parms, filter,  ds, verbose, maskGrid.get());
            break;
    }
}


////////////////////////////////////////

// Filter operations

template<typename FilterT>
inline void
SOP_OpenVDB_Filter_Level_Set::offset(const FilterParms&, FilterT& filter,
    const float offset, bool verbose, const typename FilterT::MaskType* mask)
{

    if (verbose) {
        std::cout << "Morphological " << (offset>0 ? "erosion" : "dilation")
            << " by the offset " << offset << std::endl;
    }

    filter.offset(offset, mask);
}

template<typename FilterT>
void
SOP_OpenVDB_Filter_Level_Set::mean(const FilterParms& parms, FilterT& filter,
    BossT& boss, bool verbose, const typename FilterT::MaskType* mask)
{
    for (int n = 0, N = parms.mIterations; n < N && !boss.wasInterrupted(); ++n) {

        if (verbose) {
            std::cout << "Mean filter of radius " <<  parms.mStencilWidth << std::endl;
        }

        filter.mean(parms.mStencilWidth, mask);
    }
}

template<typename FilterT>
void
SOP_OpenVDB_Filter_Level_Set::gaussian(const FilterParms& parms, FilterT& filter,
    BossT& boss, bool verbose, const typename FilterT::MaskType* mask)
{
    for (int n = 0, N = parms.mIterations; n < N && !boss.wasInterrupted(); ++n) {

        if (verbose) {
            std::cout << "Gaussian filter of radius " <<  parms.mStencilWidth << std::endl;
        }

        filter.gaussian(parms.mStencilWidth, mask);
    }
}

template<typename FilterT>
void
SOP_OpenVDB_Filter_Level_Set::median(const FilterParms& parms, FilterT& filter,
    BossT& boss, bool verbose, const typename FilterT::MaskType* mask)
{
    for (int n = 0, N = parms.mIterations; n < N && !boss.wasInterrupted(); ++n) {

        if (verbose) {
            std::cout << "Median filter of radius " << parms.mStencilWidth << std::endl;
        }

        filter.median(parms.mStencilWidth, mask);
    }
}

template<typename FilterT>
void
SOP_OpenVDB_Filter_Level_Set::meanCurvature(const FilterParms& parms, FilterT& filter,
    BossT& boss, bool verbose, const typename FilterT::MaskType* mask)
{
    for (int n = 0, N = parms.mIterations; n < N && !boss.wasInterrupted(); ++n) {

        if (verbose) std::cout << "Mean-curvature flow" << (n+1) << std::endl;

        filter.meanCurvature(mask);
    }
}

template<typename FilterT>
void
SOP_OpenVDB_Filter_Level_Set::laplacian(const FilterParms& parms, FilterT& filter,
    BossT& boss, bool verbose, const typename FilterT::MaskType* mask)
{
    for (int n = 0, N = parms.mIterations; n < N && !boss.wasInterrupted(); ++n) {

        if (verbose) std::cout << "Laplacian flow" << (n+1) << std::endl;

        filter.laplacian(mask);
    }
}

template<typename FilterT>
inline void
SOP_OpenVDB_Filter_Level_Set::renormalize(const FilterParms& parms, FilterT& filter,
    BossT& boss, bool verbose)
{
    // We will restore the old normCount since it is important to level set tracking
    const int oldNormCount = filter.getNormCount();
    filter.setNormCount(1); // only one normalization per iteration

    if (verbose) std::cout << "Renormalize #" << parms.mIterations << std::endl;

    for (int n = 0, N = parms.mIterations; n < N && !boss.wasInterrupted(); ++n) {
        filter.normalize();
    }

    filter.setNormCount(oldNormCount);
}

template<typename FilterT>
inline void
SOP_OpenVDB_Filter_Level_Set::resizeNarrowBand(const FilterParms& parms, FilterT& filter,
    BossT&, bool verbose)
{
    // We will restore the old normCount since it is important to level set tracking
    const int oldNormCount = filter.getNormCount();
    filter.setNormCount(1); // only one normalization per iteration

    const typename FilterT::ValueType gamma = filter.grid().background();
    const int wOld = static_cast<int>(openvdb::math::Round(gamma/mVoxelSize));
    const int wNew = parms.mHalfWidth;

    if (wOld < wNew) {
        if (verbose) std::cout << "Dilate narrow band #" << (wNew - wOld) << std::endl;
        filter.dilate(wNew - wOld);
    } else if (wOld > wNew) {
        if (verbose) std::cout << "Erode narrow band #" << (wOld - wNew) << std::endl;
        filter.erode(wOld - wNew);
    }

    filter.setNormCount(oldNormCount);
}

template<typename FilterT>
inline void
SOP_OpenVDB_Filter_Level_Set::track(const FilterParms& parms, FilterT& filter,
    BossT& boss, bool verbose)
{
    for (int n = 0, N = parms.mIterations; n < N && !boss.wasInterrupted(); ++n) {

        if (verbose) std::cout << "Tracking #" << (n+1) << std::endl;
        filter.track();
    }
}

////////////////////////////////////////


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
