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
/// @file SOP_OpenVDB_Filter.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Filtering operations for non-level-set grids

#include <houdini_utils/OP_NodeChain.h> // for getNodeChain(), OP_EvalScope
#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/Filter.h>
#include <OP/OP_AutoLockInputs.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_Version.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Filter: public hvdb::SOP_NodeVDB
{
public:
    // Operations should be numbered sequentially starting from 0.
    // When adding an item to the end of this list, be sure to update NUM_OPERATIONS.
    enum Operation {
        OP_MEAN = 0,
        OP_GAUSS,
        OP_MEDIAN,
#ifndef SESI_OPENVDB
        OP_OFFSET,
#endif
        NUM_OPERATIONS
    };

    static Operation intToOp(int);
    static Operation stringToOp(const std::string&);
    static std::string opToString(Operation);
    static std::string opToMenuName(Operation);


    SOP_OpenVDB_Filter(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Filter() override {}

    static void registerSop(OP_OperatorTable*);
    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned input) const override { return (input == 1); }

protected:
    struct FilterParms {
        FilterParms(Operation _op)
            : op(_op)
            , iterations(1)
            , radius(1)
            , worldRadius(0.1f)
            , minMask(0.0f)
            , maxMask(0.0f)
            , invertMask(false)
            , useWorldRadius(false)
            , mask(nullptr)
#ifndef SESI_OPENVDB
            , offset(0.0)
            , verbose(false)
#endif
        {
        }
        Operation op;
        int iterations;
        int radius;
        float worldRadius;
        float minMask, maxMask;
        bool  invertMask, useWorldRadius;
        const openvdb::FloatGrid* mask;
#ifndef SESI_OPENVDB
        float offset;
        bool verbose;
#endif
    };
    using FilterParmVec = std::vector<FilterParms>;

    OP_ERROR evalFilterParms(OP_Context&, GU_Detail&, FilterParmVec&);

    OP_ERROR cookMySop(OP_Context&) override;
    bool updateParmsFlags() override;

private:
    struct FilterOp;
};


////////////////////////////////////////


SOP_OpenVDB_Filter::Operation
SOP_OpenVDB_Filter::intToOp(int i)
{
    switch (i) {
#ifndef SESI_OPENVDB
        case OP_OFFSET: return OP_OFFSET;
#endif
        case OP_MEAN:   return OP_MEAN;
        case OP_GAUSS:  return OP_GAUSS;
        case OP_MEDIAN: return OP_MEDIAN;
        case NUM_OPERATIONS: break;
    }
    std::ostringstream ostr;
    ostr << "unknown operation (" << i << ")";
    throw std::runtime_error(ostr.str().c_str());
}


SOP_OpenVDB_Filter::Operation
SOP_OpenVDB_Filter::stringToOp(const std::string& s)
{
    if (s == "mean")   return OP_MEAN;
    if (s == "gauss")  return OP_GAUSS;
    if (s == "median") return OP_MEDIAN;
#ifndef SESI_OPENVDB
    if (s == "offset") return OP_OFFSET;
#endif
    std::ostringstream ostr;
    ostr << "unknown operation \"" << s << "\"";
    throw std::runtime_error(ostr.str().c_str());
}


std::string
SOP_OpenVDB_Filter::opToString(Operation op)
{
    switch (op) {
#ifndef SESI_OPENVDB
        case OP_OFFSET: return "offset";
#endif
        case OP_MEAN:   return "mean";
        case OP_GAUSS:  return "gauss";
        case OP_MEDIAN: return "median";
        case NUM_OPERATIONS: break;
    }
    std::ostringstream ostr;
    ostr << "unknown operation (" << int(op) << ")";
    throw std::runtime_error(ostr.str().c_str());
}


std::string
SOP_OpenVDB_Filter::opToMenuName(Operation op)
{
    switch (op) {
#ifndef SESI_OPENVDB
        case OP_OFFSET: return "Offset";
#endif
        case OP_MEAN:   return "Mean Value";
        case OP_GAUSS:  return "Gaussian";
        case OP_MEDIAN: return "Median Value";
        case NUM_OPERATIONS: break;
    }
    std::ostringstream ostr;
    ostr << "Unknown operation (" << int(op) << ")";
    throw std::runtime_error(ostr.str().c_str());
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Filter::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Filter(net, name, op);
}


SOP_OpenVDB_Filter::SOP_OpenVDB_Filter(OP_Network* net, const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


void
newSopOperator(OP_OperatorTable* table)
{
    SOP_OpenVDB_Filter::registerSop(table);
}


void
SOP_OpenVDB_Filter::registerSop(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    // Input group
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of the input VDB grids to be processed.")
        .setDocumentation(
            "A subset of the input VDBs to be processed"
            " (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "mask", "")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setTooltip("Enable / disable the mask."));

    parms.add(hutil::ParmFactory(PRM_STRING, "maskname", "Alpha Mask")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setTooltip("Optional scalar VDB used for alpha masking\n\n"
            "Values are assumed to be between 0 and 1."));

    // Menu of operations
    {
        std::vector<std::string> items;
        for (int i = 0; i < NUM_OPERATIONS; ++i) {
            const Operation op = intToOp(i);
            items.push_back(opToString(op)); // token
            items.push_back(opToMenuName(op)); // label
        }
        parms.add(hutil::ParmFactory(PRM_STRING, "operation", "Operation")
            .setDefault(opToString(OP_MEAN))
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("The operation to be applied to input volumes")
            .setDocumentation("\
The operation to be applied to input volumes\n\n\
Gaussian:\n\
    Set the value of each active voxel to a Gaussian-weighted sum\n\
    over the voxel's neighborhood.\n\n\
    This is equivalent to a Gaussian blur.\n\
Mean Value:\n\
    Set the value of each active voxel to the average value over\n\
    the voxel's neighborhood.\n\n\
    One iteration is equivalent to a box blur.  For a cone blur,\n\
    multiply the radius by 0.454545 and use two iterations.\n\
Median Value:\n\
    Set the value of each active voxel to the median value over\n\
    the voxel's neighborhood.\n\n\
    This is useful for suppressing outlier values.\n\
Offset:\n\
    Add a given offset to each active voxel's value.\n\
"));
    }

    // Filter radius

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "worldunits", "Use World Space Radius Units")
        .setTooltip(
            "If enabled, specify the filter neighborhood size in world units,\n"
            "otherwise specify the size in voxels."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "radius", "Filter Voxel Radius")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 5)
        .setDocumentation(nullptr));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "worldradius", "Filter Radius")
        .setDefault(0.1)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 10)
        .setTooltip("Half the width of a side of the cubic filter neighborhood"));

    // Number of iterations
    parms.add(hutil::ParmFactory(PRM_INT_J, "iterations", "Iterations")
        .setDefault(PRMfourDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setTooltip("The number of times to apply the operation"));

#ifndef SESI_OPENVDB
    // Offset
    parms.add(hutil::ParmFactory(PRM_FLT_J, "offset", "Offset")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_UI, -10.0, PRM_RANGE_UI, 10.0)
        .setTooltip("When the operation is Offset, add this value to all active voxels."));
#endif

     //Invert mask.
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "invert", "Invert Alpha Mask")
        .setTooltip("Invert the mask so that alpha value 0 maps to 1 and 1 to 0."));

    // Min mask range
    parms.add(hutil::ParmFactory(PRM_FLT_J, "minMask", "Min Mask Cutoff")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_UI, 0.0, PRM_RANGE_UI, 1.0)
        .setTooltip("Threshold below which mask values are clamped to zero"));

    // Max mask range
    parms.add(hutil::ParmFactory(PRM_FLT_J, "maxMask", "Max Mask Cutoff")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_UI, 0.0, PRM_RANGE_UI, 1.0)
        .setTooltip("Threshold above which mask values are clamped to one"));

#ifndef SESI_OPENVDB
    // Verbosity toggle.
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "verbose", "Verbose")
        .setTooltip("Print the sequence of operations to the terminal."));
#endif

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR,"sep1", ""));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Filter", SOP_OpenVDB_Filter::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("VDBs to Smooth")
        .addOptionalInput("Optional VDB Alpha Mask")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Filters/smooths the values in a VDB volume.\"\"\"\n\
\n\
@overview\n\
\n\
This node assigns to each active voxel in a VDB volume a value,\n\
such as the mean or median, that is representative of the voxel's neighborhood,\n\
where the neighborhood is a cube centered on the voxel.\n\
This has the effect of reducing high-frequency content and suppressing noise.\n\
\n\
If the optional scalar mask volume is provided, the output value of\n\
each voxel is a linear blend between its input value and the neighborhood value.\n\
A mask value of zero leaves the input value unchanged.\n\
\n\
NOTE:\n\
    To filter a level set, use the\n\
    [OpenVDB Smooth Level Set|Node:sop/DW_OpenVDBSmoothLevelSet] node.\n\
\n\
@related\n\
- [OpenVDB Noise|Node:sop/DW_OpenVDBNoise]\n\
- [OpenVDB Smooth Level Set|Node:sop/DW_OpenVDBSmoothLevelSet]\n\
- [Node:sop/vdbsmooth]\n\
- [Node:sop/vdbsmoothsdf]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


// Disable UI Parms.
bool
SOP_OpenVDB_Filter::updateParmsFlags()
{
    bool changed = false, hasMask = (this->nInputs() == 2);

    changed |= enableParm("mask", hasMask);
    bool useMask = bool(evalInt("mask", 0, 0)) && hasMask;
    changed |= enableParm("invert", useMask);
    changed |= enableParm("minMask", useMask);
    changed |= enableParm("maxMask", useMask);
    changed |= enableParm("maskname", useMask);

    const bool worldUnits = bool(evalInt("worldunits", 0, 0));

#ifndef SESI_OPENVDB
    UT_String s;
    evalString(s, "operation", 0, 0);

    Operation op = OP_MEAN;
    bool gotOp = false;
    try { op = stringToOp(s.toStdString()); gotOp = true; }
    catch (std::runtime_error&) {}

    // Disable and hide unused parameters.
    if (gotOp) {
        bool enable = (op == OP_MEAN || op == OP_GAUSS || op == OP_MEDIAN);
        changed |= enableParm("iterations", enable);
        changed |= enableParm("radius", enable);
        changed |= enableParm("worldradius", enable);
        changed |= setVisibleState("iterations", enable);
        changed |= setVisibleState("worldunits", enable);
        changed |= setVisibleState("radius", enable && !worldUnits);
        changed |= setVisibleState("worldradius", enable && worldUnits);

        enable = (op == OP_OFFSET);
        changed |= enableParm("offset", enable);
        changed |= setVisibleState("offset", enable);
    }
#else
    changed |= setVisibleState("radius", !worldUnits);
    changed |= setVisibleState("worldradius", worldUnits);
#endif

    return changed;
}


////////////////////////////////////////


// Helper class for use with UTvdbProcessTypedGrid()
struct SOP_OpenVDB_Filter::FilterOp
{
    FilterParmVec opSequence;
    hvdb::Interrupter* interrupt;

    template<typename GridT>
    void operator()(GridT& grid)
    {
        using ValueT = typename GridT::ValueType;
        using MaskT = openvdb::FloatGrid;

        openvdb::tools::Filter<GridT, MaskT, hvdb::Interrupter> filter(grid, interrupt);

        for (size_t i = 0, N = opSequence.size(); i < N; ++i) {
            if (interrupt && interrupt->wasInterrupted()) return;

            const FilterParms& parms = opSequence[i];

            int radius = parms.radius;

            if (parms.useWorldRadius) {
                double voxelRadius = double(parms.worldRadius) / grid.voxelSize()[0];
                radius = std::max(1, int(voxelRadius));
            }

            filter.setMaskRange(parms.minMask, parms.maxMask);
            filter.invertMask(parms.invertMask);

            switch (parms.op) {
#ifndef SESI_OPENVDB
            case OP_OFFSET:
                {
                    const ValueT offset = static_cast<ValueT>(parms.offset);
                    if (parms.verbose) std::cout << "Applying Offset by " << offset << std::endl;
                    filter.offset(offset, parms.mask);
                }
                break;
#endif
            case OP_MEAN:
#ifndef SESI_OPENVDB
                if (parms.verbose) {
                    std::cout << "Applying " << parms.iterations << " iterations of mean value"
                        " filtering with a radius of " << radius << std::endl;
                }
#endif
                filter.mean(radius, parms.iterations, parms.mask);
                break;

            case OP_GAUSS:
#ifndef SESI_OPENVDB
                if (parms.verbose) {
                    std::cout << "Applying " << parms.iterations << " iterations of gaussian"
                        " filtering with a radius of " <<radius << std::endl;
                }
#endif
                filter.gaussian(radius, parms.iterations, parms.mask);
                break;

            case OP_MEDIAN:
#ifndef SESI_OPENVDB
                if (parms.verbose) {
                    std::cout << "Applying " << parms.iterations << " iterations of median value"
                        " filtering with a radius of " << radius << std::endl;
                }
#endif
                filter.median(radius, parms.iterations, parms.mask);
                break;

            case NUM_OPERATIONS:
                break;
            }
        }
    }
};


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Filter::evalFilterParms(OP_Context& context, GU_Detail&, FilterParmVec& parmVec)
{
    hutil::OP_EvalScope evalScope(*this, context);
    const fpreal now = context.getTime();

    UT_String s;
    evalString(s, "operation", 0, 0);
    const Operation op = stringToOp(s.toStdString());

    FilterParms parms(op);
    parms.radius = static_cast<int>(evalInt("radius", 0, now));
    parms.worldRadius = float(evalFloat("worldradius", 0, now));
    parms.useWorldRadius = bool(evalInt("worldunits", 0, now));
    parms.iterations = static_cast<int>(evalInt("iterations", 0, now));
#ifndef SESI_OPENVDB
    parms.offset = static_cast<float>(evalFloat("offset", 0, now));
    parms.verbose = bool(evalInt("verbose", 0, now));
#endif
    openvdb::FloatGrid::ConstPtr maskGrid;
    if (this->nInputs() == 2 && evalInt("mask", 0, now)) {
        OP_Node* maskInputNode = getInput(1, /*mark_used*/true);
        UT_String str;
        evalString(str, "maskname", 0, now);
        std::string maskName = str.toStdString();
        GU_DetailHandle maskHandle;
        maskHandle = static_cast<SOP_Node*>(maskInputNode)->getCookedGeoHandle(context);

        GU_DetailHandleAutoReadLock maskScope(maskHandle);
        const GU_Detail *maskGeo = maskScope.getGdp();

        if (maskGeo) {
#if (UT_MAJOR_VERSION_INT >= 15)
            const GA_PrimitiveGroup * maskGroup =
                parsePrimitiveGroups(maskName.c_str(), GroupCreator(maskGeo));
#else
            const GA_PrimitiveGroup * maskGroup =
                parsePrimitiveGroups(maskName.c_str(), const_cast<GU_Detail*>(maskGeo));
#endif
            if (!maskGroup && !maskName.empty()) {
                addWarning(SOP_MESSAGE, "Mask not found.");
            } else {
                hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);
                if (maskIt) {
                    if (maskIt->getStorageType() == UT_VDB_FLOAT) {
                        maskGrid = openvdb::gridConstPtrCast<openvdb::FloatGrid>(
                            maskIt->getGridPtr());
                    } else {
                        addWarning(SOP_MESSAGE, "The mask grid has to be a FloatGrid.");
                    }
                } else {
                    addWarning(SOP_MESSAGE, "The mask input is empty.");
                }
            }
        }
    }
    parms.mask = maskGrid.get();
    parms.minMask      = static_cast<float>(evalFloat("minMask", 0, now));
    parms.maxMask      = static_cast<float>(evalFloat("maxMask", 0, now));
    parms.invertMask   = evalInt("invert", 0, now);

    parmVec.push_back(parms);

    return error();
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Filter::cookMySop(OP_Context& context)
{
    try {
        OP_AutoLockInputs lock;

        const fpreal now = context.getTime();

        FilterOp filterOp;

        SOP_OpenVDB_Filter* startNode = this;
        {
            // Find adjacent, upstream nodes of the same type as this node.
            std::vector<SOP_OpenVDB_Filter*> nodes = hutil::getNodeChain(context, this);

            startNode = nodes[0];

            // Collect filter parameters starting from the topmost node.
            FilterParmVec& parmVec = filterOp.opSequence;
            parmVec.reserve(nodes.size());
            for (size_t n = 0, N = nodes.size(); n < N; ++n) {
                if (nodes[n]->evalFilterParms(context, *gdp, parmVec) >= UT_ERROR_ABORT) {
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
        if (startNode->duplicateSourceStealable(0, context, &gdp, myGdpHandle, /*clean = */ true) >= UT_ERROR_ABORT) {
            return error();
        }

        // Get the group of grids to process.
        UT_String groupStr;
        evalString(groupStr, "group", 0, now);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        hvdb::Interrupter progress("Filtering VDB grids");
        filterOp.interrupt = &progress;

        // Process each VDB primitive in the selected group.
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {

            if (progress.wasInterrupted()) {
                throw std::runtime_error("processing was interrupted");
            }

            GU_PrimVDB* vdbPrim = *it;
            UT_String name = it.getPrimitiveNameOrIndex();

#ifndef SESI_OPENVDB
            if (evalInt("verbose", 0, now)) {
                std::cout << "\nFiltering \"" << name << "\"" << std::endl;
            }
#endif

            int success = GEOvdbProcessTypedGridTopology(*vdbPrim, filterOp);

            if (!success) {
                std::stringstream ss;
                ss << "VDB grid " << name << " of type "
                    << vdbPrim->getConstGrid().valueType() << " was skipped";
                addWarning(SOP_MESSAGE, ss.str().c_str());
                continue;
            }
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
