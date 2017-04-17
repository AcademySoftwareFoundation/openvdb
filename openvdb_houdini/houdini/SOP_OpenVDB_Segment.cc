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
/// @file SOP_OpenVDB_Segment.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Segment VDB Grids

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/Utils.h>

#include <openvdb/tools/LevelSetUtil.h>


#include <GA/GA_ElementGroupTable.h>
#include <GA/GA_PageHandle.h>
#include <GA/GA_PageIterator.h>
#include <GA/GA_AttributeInstanceMatrix.h>
#include <GEO/GEO_PrimClassifier.h>
#include <GEO/GEO_PointClassifier.h>
#include <GU/GU_ConvertParms.h>
#include <UT/UT_Quaternion.h>
#include <UT/UT_ScopedPtr.h>
#include <UT/UT_ValArray.h>
#include <UT/UT_Version.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/math/constants/constants.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <vector>
#include <list>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


namespace
{

struct SegmentActiveVoxels
{
    SegmentActiveVoxels(GU_Detail& geo, bool visualize, bool appendNumber, hvdb::Interrupter&)
        : mGeoPt(&geo)
        , mVisualize(visualize)
        , mAppendNumber(appendNumber)
    {
    }

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        using GridPtrType = typename GridType::Ptr;

        std::vector<GridPtrType> segments;

        openvdb::tools::segmentActiveVoxels(grid, segments);

        GA_RWHandleV3 color;
        if (mVisualize) {
            GA_RWAttributeRef attrRef = mGeoPt->findDiffuseAttribute(GA_ATTRIB_PRIMITIVE);
            if (!attrRef.isValid()) attrRef = mGeoPt->addDiffuseAttribute(GA_ATTRIB_PRIMITIVE);
            color.bind(attrRef.getAttribute());
        }

        float r, g, b;

        for (size_t n = 0, N = segments.size(); n < N; ++n) {

            std::string name = grid.getName();
            if (mAppendNumber) {
                std::stringstream ss;
                ss << name << "_" << n;
                name = ss.str();
            }

            GU_PrimVDB* vdb = hvdb::createVdbPrimitive(*mGeoPt, segments[n], name.c_str());
            if (color.isValid()) {
                GA_Offset offset = vdb->getMapOffset();
                exint colorID = exint(offset);
                UT_Color::getUniqueColor(colorID, &r, &g, &b);
                color.set(vdb->getMapOffset(), UT_Vector3(r, g, b));
            }
        }
    }

private:
    GU_Detail         * const mGeoPt;
    bool                const mVisualize;
    bool                const mAppendNumber;
}; // struct SegmentActiveVoxels


struct SegmentSDF
{
    SegmentSDF(GU_Detail& geo, bool visualize, bool appendNumber, hvdb::Interrupter&)
        : mGeoPt(&geo)
        , mVisualize(visualize)
        , mAppendNumber(appendNumber)
    {
    }

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        using GridPtrType = typename GridType::Ptr;

        std::vector<GridPtrType> segments;

        openvdb::tools::segmentSDF(grid, segments);

        GA_RWHandleV3 color;
        if (mVisualize) {
            GA_RWAttributeRef attrRef = mGeoPt->findDiffuseAttribute(GA_ATTRIB_PRIMITIVE);
            if (!attrRef.isValid()) attrRef = mGeoPt->addDiffuseAttribute(GA_ATTRIB_PRIMITIVE);
            color.bind(attrRef.getAttribute());
        }

        float r, g, b;

        for (size_t n = 0, N = segments.size(); n < N; ++n) {

            std::string name = grid.getName();
            if (mAppendNumber) {
                std::stringstream ss;
                ss << name << "_" << n;
                name = ss.str();
            }

            GU_PrimVDB* vdb = hvdb::createVdbPrimitive(*mGeoPt, segments[n], name.c_str());

            if (color.isValid()) {
                GA_Offset offset = vdb->getMapOffset();
                exint colorID = exint(offset);
                UT_Color::getUniqueColor(colorID, &r, &g, &b);
                color.set(offset, UT_Vector3(r, g, b));
            }
        }
    }

private:
    GU_Detail         * const mGeoPt;
    bool                const mVisualize;
    bool                const mAppendNumber;
}; // struct SegmentSDF


} // unnamed namespace


////////////////////////////////////////


class SOP_OpenVDB_Segment: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Segment(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Segment() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned i ) const override { return (i > 0); }

protected:
    OP_ERROR cookMySop(OP_Context&) override;
    bool updateParmsFlags() override;
};

////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Select a subset of the input OpenVDB grids to segment.")
        .setDocumentation(
            "A subset of the input VDB grids to be segmented"
            " (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "colorsegments", "Color Segments")
        .setDefault(PRMoneDefaults)
        .setDocumentation(
            "If enabled, assign a unique, random color to each segment"
            " for ease of identification."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "appendnumber", "Append Segment Number to Grid Name")
        .setDefault(PRMoneDefaults)
        .setDocumentation(
            "If enabled, name each output VDB after the input VDB with"
            " a unique segment number appended for ease of identification."));

    hvdb::OpenVDBOpFactory("OpenVDB Segment", SOP_OpenVDB_Segment::factory, parms, *table)
        .addInput("OpenVDB grids")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Split SDF VDB volumes into connected components.\"\"\"\n\
\n\
@overview\n\
\n\
A single SDF VDB may represent multiple disjoint objects.\n\
This node detects disjoint components and creates a new VDB for each component.\n\
\n\
@related\n\
- [Node:sop/vdbsegmentbyconnectivity]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Segment::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Segment(net, name, op);
}


SOP_OpenVDB_Segment::SOP_OpenVDB_Segment(OP_Network* net,
    const char* name, OP_Operator* op): hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


bool
SOP_OpenVDB_Segment::updateParmsFlags()
{
    bool changed = false;
    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Segment::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        gdp->clearAndDestroy();

        const fpreal time = context.getTime();

        const GU_Detail* inputGeoPt = inputGeo(0);
        const GA_PrimitiveGroup *group = nullptr;

        hvdb::Interrupter boss("VDB Segment");

        {
            UT_String str;
            evalString(str, "group", 0, time);
            group = matchGroup(const_cast<GU_Detail&>(*inputGeoPt), str.toStdString());
        }


        hvdb::VdbPrimCIterator vdbIt(inputGeoPt, group);

        if (!vdbIt) {
            addWarning(SOP_MESSAGE, "No VDB grids to process.");
            return error();
        }

        bool visualize = bool(evalInt("colorsegments", 0, time));
        bool appendNumber = bool(evalInt("appendnumber", 0, time));

        SegmentActiveVoxels segmentActiveVoxels(*gdp, visualize, appendNumber, boss);
        SegmentSDF segmentSDF(*gdp, visualize, appendNumber, boss);

        for (; vdbIt; ++vdbIt) {

            if (boss.wasInterrupted()) break;

            const GU_PrimVDB* vdb = vdbIt.getPrimitive();

            const openvdb::GridClass gridClass = vdb->getGrid().getGridClass();
            if (gridClass == openvdb::GRID_LEVEL_SET) {
                GEOvdbProcessTypedGridScalar(*vdb, segmentSDF);
            } else {
                GEOvdbProcessTypedGridTopology(*vdb, segmentActiveVoxels);
            }
        }

        if (boss.wasInterrupted()) {
            addWarning(SOP_MESSAGE, "Process was interrupted");
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
