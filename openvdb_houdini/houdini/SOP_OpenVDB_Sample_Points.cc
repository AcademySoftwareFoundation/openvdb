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
/// @file SOP_OpenVDB_Sample_Points.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Samples OpenVDB grid values as attributes on spatially located particles.
/// Currently the grid values can be scalar (float, double) or vec3 (float, double)
/// but the attributes on the particles are single precision scalar or vec3

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/Interpolation.h>  // for box sampler
#include <tbb/tick_count.h>                 // for timing
#include <tbb/task.h>                       // for cancel
#include <UT/UT_Interrupt.h>
#include <GA/GA_PageHandle.h>
#include <GA/GA_PageIterator.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;
namespace cvdb = openvdb;


class SOP_OpenVDB_Sample_Points: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Sample_Points(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Sample_Points() override {}

    static OP_Node* factory(OP_Network*, const char*, OP_Operator*);

    // The VDB port holds read-only VDBs.
    int isRefInput(unsigned input) const override { return (input == 1); }

protected:
    OP_ERROR cookMySop(OP_Context&) override;

private:
    void sample(OP_Context&);

    bool mVerbose = false;
};


////////////////////////////////////////


namespace {  // anon namespace for the sampler

template <bool staggered = false>
struct BoxSampler {
    template <class Accessor>
    static bool sample(const Accessor& in, const cvdb::Vec3R& inCoord,
        typename Accessor::ValueType& result)
    {
        return cvdb::tools::BoxSampler::sample<Accessor>(in, inCoord, result);
    }
};

template<>
struct BoxSampler<true> {
    template <class Accessor>
    static bool sample(const Accessor& in, const cvdb::Vec3R& inCoord,
        typename Accessor::ValueType& result)
    {
        return cvdb::tools::StaggeredBoxSampler::sample<Accessor>(in, inCoord, result);
    }
};

template <bool staggered = false>
struct NearestNeighborSampler {
    template <class Accessor>
    static bool sample(const Accessor& in, const cvdb::Vec3R& inCoord,
        typename Accessor::ValueType& result)
    {
        return cvdb::tools::PointSampler::sample<Accessor>(in, inCoord, result);
    }
};

template<>
struct NearestNeighborSampler<true> {
    template <class Accessor>
    static bool sample(const Accessor& in, const cvdb::Vec3R& inCoord,
        typename Accessor::ValueType& result)
    {
        return cvdb::tools::StaggeredPointSampler::sample<Accessor>(in, inCoord, result);
    }
};


template<
    typename GridType,
    typename GA_RWPageHandleType,
    bool staggered = false,
    bool NearestNeighbor = false>
class PointSampler
{
public:
    using Accessor = typename GridType::ConstAccessor;

    // constructor. from grid and GU_Detail*
    PointSampler(const hvdb::Grid& grid, const bool threaded,
                 GU_Detail* gdp, GA_RWAttributeRef& handle,
                 UT_AutoInterrupt* interrupter):
        mGrid(grid),
        mThreaded(threaded),
        mGdp(gdp),
        mAttribPageHandle(handle.getAttribute()),
        mInterrupter(interrupter)
    {
    }

    // constructor.  from other
    PointSampler(const PointSampler<GridType, GA_RWPageHandleType, staggered>& other):
        mGrid(other.mGrid),
        mThreaded(other.mThreaded),
        mGdp(other.mGdp),
        mAttribPageHandle(other.mAttribPageHandle),
        mInterrupter(other.mInterrupter)
    {
    }

    void sample()
    {
        if (mThreaded) {
            // multi-threaded
            UTparallelFor(GA_SplittableRange(mGdp->getPointRange()), *this);
        } else {
            // single-threaded
            (*this)(GA_SplittableRange(mGdp->getPointRange()));
        }
    }

    // only the supported versions don't throw
    void operator() (const GA_SplittableRange& range) const
    {

        if (mInterrupter->wasInterrupted()) {
            tbb::task::self().cancel_group_execution();
        }
        const GridType& grid = UTvdbGridCast<GridType>(mGrid);
        // task local grid accessor
        Accessor accessor = grid.getAccessor();
        // sample scalar data onto points
        typename GridType::ValueType value;
        cvdb::Vec3R point;

        GA_ROPageHandleV3   p_ph(mGdp->getP());
        GA_RWPageHandleType v_ph = mAttribPageHandle;

        if(!v_ph.isValid()) {
            throw std::runtime_error("new attribute not valid");
        }

        // iterate over pages in the range
        for (GA_PageIterator pit = range.beginPages(); !pit.atEnd(); ++pit) {
            GA_Offset start;
            GA_Offset end;

            // per-page setup
            p_ph.setPage(*pit);
            v_ph.setPage(*pit);
            // iterate over elements in the page
            for (GA_Iterator it(pit.begin()); it.blockAdvance(start, end); ) {
                for (GA_Offset offset = start; offset < end; ++offset ) {
                    // get the pos.
                    UT_Vector3 pos = p_ph.get(offset);
                    // find the interpolated value
                    point = mGrid.worldToIndex(cvdb::Vec3R(pos[0], pos[1], pos[2]));

                    if (NearestNeighbor) {
                        NearestNeighborSampler<staggered>::template sample<Accessor>(
                            accessor, point, value);
                    } else {
                        BoxSampler<staggered>::template sample<Accessor>(accessor, point, value);
                    }
                    // set the value
                    v_ph.value(offset) = translateValue(value);
                }
            }
        }
    }
    template<typename T> inline static float translateValue(const T& vdb_value) {
        return static_cast<float>(vdb_value);
    }
    inline static UT_Vector3 translateValue(cvdb::Vec3f& vdb_value) {
        return UT_Vector3(vdb_value[0], vdb_value[1], vdb_value[2]);
    }
    inline static UT_Vector3 translateValue(cvdb::Vec3d& vdb_value) {
        return UT_Vector3(
            static_cast<float>(vdb_value[0]),
            static_cast<float>(vdb_value[1]),
            static_cast<float>(vdb_value[2]));
    }

private:
    // member data
    const hvdb::Grid&    mGrid;
    bool                 mThreaded;
    GU_Detail*           mGdp;
    GA_RWPageHandleType  mAttribPageHandle;
    UT_AutoInterrupt*    mInterrupter;
}; // class PointSampler

} // end anonymous namespace for this sampler


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    // Group pattern
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setTooltip("Specify a subset of the input VDB grids to be processed.")
        .setDocumentation(
            "A subset of the input VDBs to be processed"
            " (see [specifying volumes|/model/volumes#group])"));

    // verbose option toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "verbose", "Verbose")
        .setDefault(PRMzeroDefaults)
        .setTooltip("If enabled, print the sequence of operations to the terminal."));

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "threaded", "Multi-threading"));
    obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep2", "Separator"));

    // Register the SOP
    hvdb::OpenVDBOpFactory("OpenVDB Sample Points",
        SOP_OpenVDB_Sample_Points::factory, parms, *table)
        .addAlias("OpenVDB Point Sample")
        .setObsoleteParms(obsoleteParms)
        .addInput("Points")
        .addInput("VDB")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Sample VDB voxel values onto points.\"\"\"\n\
\n\
@overview\n\
\n\
This node samples VDB voxel values as attributes on spatially located particles.\n\
Currently, the voxel values can be single- or double-precision scalars or vectors,\n\
but the attributes on the particles will be single-precision only.\n\
\n\
@related\n\
- [OpenVDB From Particles|Node:sop/DW_OpenVDBFromParticles]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");

}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Sample_Points::factory(OP_Network* net, const char* name, OP_Operator *op)
{
    return new SOP_OpenVDB_Sample_Points(net, name, op);
}


SOP_OpenVDB_Sample_Points::SOP_OpenVDB_Sample_Points(
    OP_Network* net, const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


void
SOP_OpenVDB_Sample_Points::sample(OP_Context& context)
{
    // this is the heart of the cook
    const fpreal time = context.getTime();

    GU_Detail* aGdp = gdp; // where the points live
    const GU_Detail* bGdp = inputGeo(1, context); // where the grids live

    // extract UI data
    mVerbose = bool(evalInt("verbose", 0, time));
    const bool threaded = true; /*evalInt("threaded", 0, time);*/
    const GA_Size nPoints = aGdp->getNumPoints();

    // sanity checks
    if (nPoints == 0) {
        const std::string msg("No points found in first input port");
        addWarning(SOP_MESSAGE, msg.c_str());
        if (mVerbose) std::cout << msg << std::endl;
    }

    // Get the group of grids to process
    UT_String groupStr;
    evalString(groupStr, "group", 0, time);

    const GA_PrimitiveGroup* group =
        matchGroup(const_cast<GU_Detail&>(*bGdp), groupStr.toStdString());

    // scratch variables used in the loop
    GA_Defaults defaultFloat(0.0), defaultInt(0);

    int numScalarGrids  = 0;
    int numVectorGrids  = 0;
    int numUnnamedGrids = 0;

    // start time
    tbb::tick_count time_start = tbb::tick_count::now();
    UT_AutoInterrupt progress("Sampling from VDB grids");


    for (hvdb::VdbPrimCIterator it(bGdp, group); it; ++it) {
        if (progress.wasInterrupted()) {
            throw std::runtime_error("was interrupted");
        }

        const GU_PrimVDB* vdb = *it;
        UT_VDBType gridType = vdb->getStorageType();
        const hvdb::Grid& grid = vdb->getGrid();

        std::string gridName = it.getPrimitiveName().toStdString();
        if (gridName.empty()) {
            std::stringstream ss;
            ss << "VDB_" << numUnnamedGrids++;
            gridName = ss.str();
        }

        // remove any dot "." characters, attribute names can't contain this.
        std::replace(gridName.begin(), gridName.end(), '.', '_');

        //convert gridName to uppercase so we can use it as a local variable name
        std::string gridVariableName = gridName;
        std::transform(gridVariableName.begin(), gridVariableName.end(),
                       gridVariableName.begin(), ::toupper);

        if (gridType == UT_VDB_FLOAT || gridType == UT_VDB_DOUBLE) {
            // a grid that holds a scalar field (as either float or double type)
            // count
            numScalarGrids++;

            //find or create float attribute
            GA_RWAttributeRef attribHandle =
                aGdp->findFloatTuple(GA_ATTRIB_POINT, gridName.c_str(), 1);
            if (!attribHandle.isValid()) {
                attribHandle =
                    aGdp->addFloatTuple(GA_ATTRIB_POINT, gridName.c_str(), 1, defaultFloat);
            }
            aGdp->addVariableName(gridName.c_str(), gridVariableName.c_str());

            // user feedback
            if (mVerbose) {
                std::cout << "Sampling grid " << gridName << " of type "
                    << grid.valueType() << std::endl;
            }

            UT_AutoInterrupt scalarInterrupt("Sampling from VDB floating-type grids");
            // do the sampling
            if (gridType == UT_VDB_FLOAT) {
                // float scalar
                PointSampler<cvdb::FloatGrid, GA_RWPageHandleF> theSampler(
                    grid, threaded, aGdp, attribHandle, &scalarInterrupt);
                theSampler.sample();

            } else {
                // double scalar
                PointSampler<cvdb::DoubleGrid, GA_RWPageHandleF> theSampler(
                    grid, threaded, aGdp, attribHandle, &scalarInterrupt);
                theSampler.sample();
            }

        } else if (gridType == UT_VDB_INT32 || gridType == UT_VDB_INT64) {
            numScalarGrids++;

            //find or create integer attribute
            GA_RWAttributeRef attribHandle =
                aGdp->findIntTuple(GA_ATTRIB_POINT, gridName.c_str(), 1);
            if (!attribHandle.isValid()) {
                attribHandle =
                    aGdp->addIntTuple(GA_ATTRIB_POINT, gridName.c_str(), 1, defaultInt);
            }
            aGdp->addVariableName(gridName.c_str(), gridVariableName.c_str());

             // user feedback
            if (mVerbose) {
                std::cout << "Sampling grid " << gridName << " of type "
                    << grid.valueType() << std::endl;
            }

            UT_AutoInterrupt scalarInterrupt("Sampling from VDB integer-type grids");
            if (gridType == UT_VDB_INT32) {

                PointSampler<cvdb::Int32Grid, GA_RWPageHandleF, false, true>
                    theSampler(grid, threaded, aGdp, attribHandle, &scalarInterrupt);
                theSampler.sample();

            } else {
                PointSampler<cvdb::Int64Grid, GA_RWPageHandleF, false, true>
                    theSampler(grid, threaded, aGdp, attribHandle, &scalarInterrupt);
                theSampler.sample();
            }

        } else if (gridType == UT_VDB_VEC3F || gridType == UT_VDB_VEC3D) {
            // a grid that holds Vec3 data (as either float or double)
            // count
            numVectorGrids++;

            // find or create create vector attribute
            GA_RWAttributeRef attribHandle =
                aGdp->findFloatTuple(GA_ATTRIB_POINT, gridName.c_str(), 3);
            if (!attribHandle.isValid()) {
                attribHandle =
                    aGdp->addFloatTuple(GA_ATTRIB_POINT, gridName.c_str(), 3, defaultFloat);
            }
            aGdp->addVariableName(gridName.c_str(), gridVariableName.c_str());

            // user feedback
            if (grid.getGridClass() != openvdb::GRID_STAGGERED) {
                // regular (non-staggered) vec3 grid
                if (mVerbose) {
                    std::cout << "Sampling grid " << gridName << " of type "
                        << grid.valueType() << std::endl;
                }

                UT_AutoInterrupt vectorInterrupt("Sampling from VDB vector-type grids");
                // do the sampling
            if (gridType == UT_VDB_VEC3F) {
                    // Vec3f
                    PointSampler<cvdb::Vec3fGrid, GA_RWPageHandleV3> theSampler(
                        grid, threaded, aGdp, attribHandle, &vectorInterrupt);
                    theSampler.sample();
                } else {
                    // Vec3d
                    PointSampler<cvdb::Vec3dGrid, GA_RWPageHandleV3> theSampler(
                        grid, threaded, aGdp, attribHandle, &vectorInterrupt);
                    theSampler.sample();
                }
            } else {
                // staggered grid case
                if (mVerbose) {
                    std::cout << "Sampling staggered grid " << gridName << " of type "
                        << grid.valueType() << std::endl;
                }

                UT_AutoInterrupt vectorInterrupt("Sampling from VDB vector-type staggered grids");
                // do the sampling
                if (grid.isType<cvdb::Vec3fGrid>()) {
                    // Vec3f
                    PointSampler<cvdb::Vec3fGrid, GA_RWPageHandleV3, true> theSampler(
                        grid, threaded, aGdp, attribHandle, &vectorInterrupt);
                    theSampler.sample();
                } else {
                    // Vec3d
                    PointSampler<cvdb::Vec3dGrid, GA_RWPageHandleV3, true> theSampler(
                        grid, threaded, aGdp, attribHandle, &vectorInterrupt);
                    theSampler.sample();
                }
            }
        } else {
            std::cout << "Skipping grid " << gridName << " of unknown type" << std::endl;
        }
    }//end iter

    // timing: end time
    tbb::tick_count time_end = tbb::tick_count::now();

    if (mVerbose) {
        std::cout << "Sampling " << nPoints << " points in "
                  << numVectorGrids << " vector grid" << (numVectorGrids == 1 ? "" : "s")
                  << " and " << numScalarGrids << " scalar grid"
                      << (numScalarGrids == 1 ? "" : "s")
                  << " took " << (time_end - time_start).seconds() << " seconds\n "
                  << ( (threaded) ? "threaded" : "non-threaded") <<std::endl;
    }
} //end sample()


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Sample_Points::cookMySop(OP_Context& context)
{
    // Surround all the work in a try statement, the base class throws
    // errors as well so we can catch and handle as elegantly as possible
    try {
        hutil::ScopedInputLock lock(*this, context);

        // this does a shallow copy of the VDB-grids and a deep copy of native Houdini primitives
        // (the points we modify in this case)
        duplicateSource(0, context);

        // do the work
        sample(context);

    } catch ( std::exception& e) {
        addError(SOP_MESSAGE, e.what() );
    }

    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
