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
/// @file SOP_OpenVDB_Sort_Points.cc
///
/// @author Mihai Alden

#include <houdini_utils/ParmFactory.h>

#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/GU_VDBPointTools.h>

#include <openvdb/tools/PointPartitioner.h>

#include <UT/UT_Version.h>
#include <GU/GU_Detail.h>
#include <PRM/PRM_Parm.h>
#include <GA/GA_SplittableRange.h>
#include <GA/GA_PageIterator.h>
#include <GA/GA_ElementWrangler.h>
#include <GA/GA_AttributeFilter.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <boost/scoped_array.hpp>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////

// Local Utility Methods

namespace {

struct CopyElements {

    CopyElements(GA_PointWrangler& wrangler, const GA_Offset* offsetArray)
        : mWrangler(&wrangler), mOffsetArray(offsetArray) { }

    void operator()(const GA_SplittableRange& range) const {
        GA_Offset start, end;
        for (GA_PageIterator pageIt = range.beginPages(); !pageIt.atEnd(); ++pageIt) {
            for (GA_Iterator blockIt(pageIt.begin()); blockIt.blockAdvance(start, end); ) {
                for (GA_Offset i = start; i < end; ++i) {
                    mWrangler->copyAttributeValues(i, mOffsetArray[i]);
                }
            }
        }
    }

    GA_PointWrangler          * const mWrangler;
    GA_Offset           const * const mOffsetArray;
}; // struct CopyElements


struct SetOffsets
{
    using PointPartitioner = openvdb::tools::UInt32PointPartitioner;

    SetOffsets(const GU_Detail& srcGeo, const PointPartitioner& partitioner, GA_Offset* offsetArray)
        : mSrcGeo(&srcGeo), mPartitioner(&partitioner), mOffsetArray(offsetArray) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        size_t idx = 0;
        for (size_t n = 0, N = range.begin(); n != N; ++n) {
            idx += mPartitioner->indices(n).size(); // increment to start index
        }

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            for (PointPartitioner::IndexIterator it = mPartitioner->indices(n); it; ++it) {
                mOffsetArray[idx++] = mSrcGeo->pointOffset(*it);
            }
        }
    }

    GU_Detail           const * const mSrcGeo;
    PointPartitioner    const * const mPartitioner;
    GA_Offset                 * const mOffsetArray;
}; // struct SetOffsets

} // unnamed namespace


////////////////////////////////////////

// SOP Implementation

struct SOP_OpenVDB_Sort_Points: public hvdb::SOP_NodeVDB {

    SOP_OpenVDB_Sort_Points(OP_Network*, const char* name, OP_Operator*);
    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    OP_ERROR cookMySop(OP_Context&) override;
};

void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "pointgroup", "Point Group")
        .setChoiceList(&SOP_Node::pointGroupMenu)
        .setTooltip("A group of points to rasterize."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "binsize", "Bin Size")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 5)
        .setTooltip("The size (length of a side) of the cubic bin, in world units."));

    hvdb::OpenVDBOpFactory("OpenVDB Sort Points",
        SOP_OpenVDB_Sort_Points::factory, parms, *table)
        .addInput("points")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Reorder points into spatially-organized bins.\"\"\"\n\
\n\
@overview\n\
\n\
This node reorders Houdini points so that they are sorted into\n\
three-dimensional spatial bins.\n\
By increasing CPU cache locality of point data, sorting can improve the\n\
performance of algorithms such as rasterization that rely on neighbor access.\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}

OP_Node*
SOP_OpenVDB_Sort_Points::factory(OP_Network* net, const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Sort_Points(net, name, op);
}

SOP_OpenVDB_Sort_Points::SOP_OpenVDB_Sort_Points(OP_Network* net, const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDB(net, name, op)
{
}

OP_ERROR
SOP_OpenVDB_Sort_Points::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        gdp->stashAll();

        const fpreal time = context.getTime();
        const GU_Detail* srcGeo = inputGeo(0);

        boost::scoped_array<GA_Offset> srcOffsetArray;
        size_t numPoints = 0;

        { // partition points and construct ordered offset list
            UT_String groupStr;
            evalString(groupStr, "pointgroup", 0, time);

#if (UT_MAJOR_VERSION_INT >= 15)
            const GA_PointGroup* pointGroup =
                parsePointGroups(groupStr, GroupCreator(srcGeo));
#else
            const GA_PointGroup* pointGroup =
                parsePointGroups(groupStr, const_cast<GU_Detail*>(srcGeo));
#endif

            const fpreal voxelSize = evalFloat("binsize", 0, time);
            const openvdb::math::Transform::Ptr transform =
                openvdb::math::Transform::createLinearTransform(voxelSize);

            GU_VDBPointList<openvdb::Vec3s> points(*srcGeo, pointGroup);

            openvdb::tools::UInt32PointPartitioner partitioner;
            partitioner.construct(points, *transform, /*voxel order=*/true);

            numPoints = points.size();
            srcOffsetArray.reset(new GA_Offset[numPoints]);

            tbb::parallel_for(tbb::blocked_range<size_t>(0, partitioner.size()),
                SetOffsets(*srcGeo, partitioner, srcOffsetArray.get()));
        }

        // order point attributes

#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
        gdp->appendPointBlock(numPoints);
#else
        for (size_t n = 0; n < numPoints; ++n) gdp->appendPointOffset();
#endif

        gdp->cloneMissingAttributes(*srcGeo, GA_ATTRIB_POINT, GA_AttributeFilter::selectPublic());

        GA_PointWrangler ptWrangler(*gdp, *srcGeo,  GA_PointWrangler::INCLUDE_P);

        UTparallelFor(GA_SplittableRange(gdp->getPointRange()),
            CopyElements(ptWrangler, srcOffsetArray.get()));

        gdp->destroyStashed();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
