///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Vector_Merge.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Merge groups of up to three scalar grids into vector grids.

#ifdef _WIN32
#define BOOST_REGEX_NO_LIB
#endif

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_String.h>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/regex.hpp>
#include <set>
#include <sstream>
#include <vector>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

// HAVE_MERGE_GROUP is disabled in H12.5
#ifdef SESI_OPENVDB
#define HAVE_MERGE_GROUP 0
#else
#define HAVE_MERGE_GROUP 1
#endif


class SOP_OpenVDB_Vector_Merge: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Vector_Merge(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Vector_Merge() {};

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);

    static void addWarningMessage(SOP_OpenVDB_Vector_Merge* self, const char* msg)
        { if (self && msg) self->addWarning(SOP_MESSAGE, msg); }
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Group of X grids
    parms.add(hutil::ParmFactory(PRM_STRING, "scalar_x_group", "X Group")
        .setDefault(0, "@name=*.x")
        .setHelpText(
            "Specify a group of scalar input VDB grids to be used\n"
            "as the x components of the merged vector grids.\n"
            "Each x grid will be paired with a y and a z grid\n"
            "(if provided) to produce an output vector grid.")
        .setChoiceList(&hutil::PrimGroupMenu));

    // Group of Y grids
    parms.add(hutil::ParmFactory(PRM_STRING, "scalar_y_group", "Y Group")
        .setDefault(0, "@name=*.y")
        .setHelpText(
            "Specify a group of scalar input VDB grids to be used\n"
            "as the y components of the merged vector grids.\n"
            "Each y grid will be paired with an x and a z grid\n"
            "(if provided) to produce an output vector grid.")
        .setChoiceList(&hutil::PrimGroupMenu));

    // Group of Z grids
    parms.add(hutil::ParmFactory(PRM_STRING, "scalar_z_group", "Z Group")
        .setDefault(0, "@name=*.z")
        .setHelpText(
            "Specify a group of scalar input VDB grids to be used\n"
            "as the z components of the merged vector grids.\n"
            "Each z grid will be paired with an x and a y grid\n"
            "(if provided) to produce an output vector grid.")
        .setChoiceList(&hutil::PrimGroupMenu));

    // Use X name
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "usexname",  "Use Basename of X VDB")
#ifndef SESI_OPENVDB
        .setDefault(PRMzeroDefaults));
#else
        .setDefault(PRMoneDefaults));
#endif

    // Output vector grid name
    parms.add(hutil::ParmFactory(PRM_STRING, "merge_name",  "Merged VDB Name")
        .setDefault(0, "merged#")
        .setHelpText(
            "Specify a name for the merged vector grids.\n"
            "Include '#' in the name to number the output grids\n"
            "in the order that they are processed."));

#if HAVE_MERGE_GROUP
    // Output vector grid group name
    parms.add(hutil::ParmFactory(PRM_STRING, "group",  "Merge Group")
        .setTypeExtended(PRM_TYPE_JOIN_PAIR)
        .setHelpText("Specify a name for the output group of merged vector grids."));

    // Toggle to enable/disable grouping
    parms.add(hutil::ParmFactory(PRM_TOGGLE | PRM_Type(PRM_Type::PRM_INTERFACE_LABEL_NONE),
            "enable_grouping", "")
        .setDefault(PRMoneDefaults)
        .setHelpText("If disabled, don't group merged vector grids."));
#endif

    // Toggle to keep/remove source grids
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "remove_sources", "Remove Source VDBs")
        .setDefault(PRMoneDefaults)
        .setHelpText("Remove scalar grids that have been merged."));

    // Toggle to copy inactive values in addition to active values
    parms.add(
        hutil::ParmFactory(PRM_TOGGLE, "copyinactive", "Copy Inactive Values")
        .setDefault(PRMzeroDefaults)
        .setHelpText(
            "If enabled, merge the values of both active and inactive voxels.\n"
            "If disabled, merge the values of active voxels only, treating\n"
            "inactive voxels as active background voxels wherever\n"
            "corresponding input voxels have different active states."));

#ifndef SESI_OPENVDB
    // Verbosity toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "verbose", "Verbose"));
#endif

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Vector Merge",
        SOP_OpenVDB_Vector_Merge::factory, parms, *table)
        .addInput("Scalar VDBs to merge into vector");
}


OP_Node*
SOP_OpenVDB_Vector_Merge::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Vector_Merge(net, name, op);
}


SOP_OpenVDB_Vector_Merge::SOP_OpenVDB_Vector_Merge(OP_Network* net,
    const char* name, OP_Operator* op):
    SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


namespace {

// Mapping from scalar ValueTypes to Vec3::value_types of registered vector-valued Grid types
template<typename T> struct VecValueTypeMap { typedef T Type; static const bool Changed = false; };
//template<> struct VecValueTypeMap<bool> {
//    typedef int32_t Type; static const bool Changed = true;
//};
//template<> struct VecValueTypeMap<uint32_t> {
//    typedef int32_t Type; static const bool Changed = true;
//};
//template<> struct VecValueTypeMap<int64_t> {
//    typedef int32_t Type; static const bool Changed = true;
//};
//template<> struct VecValueTypeMap<uint64_t> {
//    typedef int32_t Type; static const bool Changed = true;
//};


class ScalarGridMerger
{
public:
    typedef boost::function<void (const char*)> WarnFunc;

    ScalarGridMerger(
        const hvdb::Grid* x, const hvdb::Grid* y, const hvdb::Grid* z,
        const std::string& outGridName, bool copyInactiveValues,
        WarnFunc warn, UT_Interrupt* interrupt = NULL):
        mOutGridName(outGridName),
        mCopyInactiveValues(copyInactiveValues),
        mWarn(warn),
        mInterrupt(interrupt)
    {
        mInGrid[0] = x; mInGrid[1] = y; mInGrid[2] = z;
    }

    const hvdb::GridPtr& getGrid() { return mOutGrid; }

    template<typename ScalarGridT>
    void operator()(const ScalarGridT& /*ignored*/)
    {
        if (!mInGrid[0] && !mInGrid[1] && !mInGrid[2]) return;

        typedef typename ScalarGridT::TreeType ScalarTreeT;

        // Retrieve a scalar tree from each input grid.
        const ScalarTreeT* inTree[3] = { NULL, NULL, NULL };
        if (mInGrid[0]) inTree[0] = &UTvdbGridCast<ScalarGridT>(mInGrid[0])->tree();
        if (mInGrid[1]) inTree[1] = &UTvdbGridCast<ScalarGridT>(mInGrid[1])->tree();
        if (mInGrid[2]) inTree[2] = &UTvdbGridCast<ScalarGridT>(mInGrid[2])->tree();
        if (!inTree[0] && !inTree[1] && !inTree[2]) return;

        // Get the type of the output vector tree.
        // 1. ScalarT is the input scalar tree's value type.
        typedef typename ScalarTreeT::ValueType                             ScalarT;
        // 2. VecT is Vec3<ScalarT>, provided that there is a registered Tree with that
        //    value type.  If not, use the closest match (e.g., vec3i when ScalarT = bool).
        typedef VecValueTypeMap<ScalarT>                                    MappedVecT;
        typedef openvdb::math::Vec3<typename MappedVecT::Type>              VecT;
        // 3. VecTreeT is the type of a tree with the same height and node dimensions
        //    as the input scalar tree, but with value type VecT instead of ScalarT.
        typedef typename ScalarTreeT::template ValueConverter<VecT>::Type   VecTreeT;
        typedef typename openvdb::Grid<VecTreeT>                            VecGridT;

        if (MappedVecT::Changed && mWarn) {
            std::ostringstream ostr;
            ostr << "grids of type vec3<" << openvdb::typeNameAsString<ScalarT>()
                << "> are not supported; using " << openvdb::typeNameAsString<VecT>()
                << " instead";
            if (!mOutGridName.empty()) ostr << " for " << mOutGridName;
            mWarn(ostr.str().c_str());
        }

        // Determine the background value and the transform.
        VecT bkgd(0, 0, 0);
        const openvdb::math::Transform* xform = NULL;
        for (int i = 0; i < 3; ++i) {
            if (inTree[i]) bkgd[i] = inTree[i]->background();
            if (mInGrid[i] && !xform) xform = &(mInGrid[i]->transform());
        }
        openvdb::math::Transform::Ptr outXform;
        if (xform) outXform = xform->copy();

        // Construct the output vector grid, with a background value whose
        // components are the background values of the input scalar grids.
        typename VecGridT::Ptr vecGrid = VecGridT::create(bkgd);
        mOutGrid = vecGrid;

        if (outXform) {
            mOutGrid->setTransform(outXform);

            // Check that all three input grids have the same transform.
            bool xformMismatch = false;
            for (int i = 0; i < 3 && !xformMismatch; ++i) {
                if (mInGrid[i]) {
                    const openvdb::math::Transform* inXform = &(mInGrid[i]->transform());
                    if (*outXform != *inXform) xformMismatch = true;
                }
            }
            if (xformMismatch && mWarn) {
                mWarn("component grids have different transforms");
            }
        }

        openvdb::CoordBBox bbox;
        VecT val;
        typename VecGridT::Accessor vecAccessor = vecGrid->getAccessor();
        // For each voxel in each of the input scalar grids (because the input grids
        // might have voxels set in different locations), set the appropriate
        // vector component of the corresponding voxel in the output vector grid.
        for (int i = 0; i < 3; ++i) {
            if (mInterrupt && mInterrupt->opInterrupt()) { mOutGrid.reset(); return; }
            if (!inTree[i]) continue;

            if (mCopyInactiveValues) {
                typename ScalarTreeT::ValueAllCIter it = inTree[i]->cbeginValueAll();
                for ( ; it; ++it) {
                    if (!it.getBoundingBox(bbox)) continue;

                    const bool active = it.isValueOn();

                    // Get the vector value at the current location in
                    // the output grid and update its ith component.
                    val = vecAccessor.getValue(bbox.min());
                    val[i] = it.getValue();

                    if (it.isTileValue()) {
                        vecGrid->fill(bbox, val, active);
                    } else { // it.isVoxelValue()
                        vecAccessor.setValue(bbox.min(), val);
                        if (!active) vecAccessor.setValueOff(bbox.min());
                    }
                }
            } else {
                typename ScalarTreeT::ValueOnCIter it = inTree[i]->cbeginValueOn();
                for ( ; it; ++it) {
                    if (!it.getBoundingBox(bbox)) continue;

                    // Get the vector value at the current location in
                    // the output grid and update its ith component.
                    val = vecAccessor.getValue(bbox.min());
                    val[i] = it.getValue();

                    if (it.isTileValue()) {
                        vecGrid->fill(bbox, val);
                    } else { // it.isVoxelValue()
                        vecAccessor.setValueOn(bbox.min(), val);
                    }
                }
            }
        }
    }

private:
    const hvdb::Grid* mInGrid[3];
    hvdb::GridPtr mOutGrid;
    std::string mOutGridName;
    bool mCopyInactiveValues;
    WarnFunc mWarn;
    UT_Interrupt* mInterrupt;
}; // class ScalarGridMerger

} // unnamed namespace


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Vector_Merge::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal time = context.getTime();

        duplicateSource(0, context);

        const bool copyInactiveValues = evalInt("copyinactive", 0, time);
        const bool removeSourceGrids = evalInt("remove_sources", 0, time);
#ifndef SESI_OPENVDB
        const bool verbose = evalInt("verbose", 0, time);
#else
        const bool verbose = false;
#endif

        // Get the name (or naming pattern) for merged grids.
        UT_String mergeName;
        evalString(mergeName, "merge_name", 0, time);
        const bool useXName = evalInt("usexname", 0, time);

#if HAVE_MERGE_GROUP
        // Get the group name for merged grids.
        UT_String mergeGroupStr;
        if (evalInt("enable_grouping", 0, time)) {
            evalString(mergeGroupStr, "group", 0, time);
        }
#endif

        UT_AutoInterrupt progress("Merging VDB grids");

        typedef std::set<GEO_PrimVDB*> PrimVDBSet;
        PrimVDBSet primsToRemove;

        // Get the groups of x, y and z scalar grids to merge.
        const GA_PrimitiveGroup *xGroup = NULL, *yGroup = NULL, *zGroup = NULL;
        {
            UT_String groupStr;
            evalString(groupStr, "scalar_x_group", 0, time);
            xGroup = matchGroup(*gdp, groupStr.toStdString());
            evalString(groupStr, "scalar_y_group", 0, time);
            yGroup = matchGroup(*gdp, groupStr.toStdString());
            evalString(groupStr, "scalar_z_group", 0, time);
            zGroup = matchGroup(*gdp, groupStr.toStdString());
        }

        typedef std::vector<GEO_PrimVDB*> PrimVDBVec;
        PrimVDBVec primsToGroup;

        // Iterate over VDB primitives in the selected groups.
        hvdb::VdbPrimIterator
            xIt(xGroup ? gdp : NULL, xGroup),
            yIt(yGroup ? gdp : NULL, yGroup),
            zIt(zGroup ? gdp : NULL, zGroup);
        for (int i = 1; xIt || yIt || zIt; ++xIt, ++yIt, ++zIt, ++i) {
            if (progress.wasInterrupted()) return error();

            GU_PrimVDB *xVdb = *xIt, *yVdb = *yIt, *zVdb = *zIt, *nonNullVdb = NULL;

            // Extract grids from the VDB primitives and find one that is non-null.
            // Process the primitives in ZYX order to ensure the X grid is preferred.
            /// @todo nonNullGrid's ValueType determines the ValueType of the
            /// output grid's vectors, so ideally nonNullGrid should be the
            /// grid with the highest-precision ValueType.
            const hvdb::Grid *xGrid = NULL, *yGrid = NULL, *zGrid = NULL, *nonNullGrid = NULL;
            if (zVdb) { zGrid = nonNullGrid = &zVdb->getGrid(); nonNullVdb = zVdb; }
            if (yVdb) { yGrid = nonNullGrid = &yVdb->getGrid(); nonNullVdb = yVdb; }
            if (xVdb) { xGrid = nonNullGrid = &xVdb->getGrid(); nonNullVdb = xVdb; }

            if (!nonNullGrid) continue;

            std::string outGridName;
            if (mergeName.isstring()) {
                UT_String s; s.itoa(i);
                outGridName = boost::regex_replace(
                    mergeName.toStdString(), boost::regex("#+"), s.toStdString());
            }

            if (useXName && nonNullVdb) {
                UT_String gridName(nonNullVdb->getGridName());
                UT_String basename = gridName.pathUpToExtension();
                if (basename.isstring()) {
                    outGridName = basename.toStdString();
                }
            }

            // Merge the input grids into an output grid.
            // This does not support a partial set so we quit early in that case.
            ScalarGridMerger op(xGrid, yGrid, zGrid, outGridName, copyInactiveValues,
                boost::bind(&SOP_OpenVDB_Vector_Merge::addWarningMessage, this, _1));
            UTvdbProcessTypedGridScalar(UTvdbGetGridType(*nonNullGrid), *nonNullGrid, op);

            if (hvdb::GridPtr outGrid = op.getGrid()) {
                outGrid->insertMeta("name", openvdb::StringMetadata(outGridName));

                if (verbose) {
                    std::ostringstream ostr;
                    ostr << "Merged ("
                        << (xVdb ? xVdb->getGridName() : "0") << ", "
                        << (yVdb ? yVdb->getGridName() : "0") << ", "
                        << (zVdb ? zVdb->getGridName() : "0") << ")";
                    if (!outGridName.empty()) ostr << " into " << outGridName;
                    addMessage(SOP_MESSAGE, ostr.str().c_str());
                }

                if (GEO_PrimVDB* outVdb = hvdb::createVdbPrimitive(*gdp, outGrid)) {
                    primsToGroup.push_back(outVdb);
                }

                // Flag the input grids for removal.
                primsToRemove.insert(xVdb);
                primsToRemove.insert(yVdb);
                primsToRemove.insert(zVdb);
            }
        }

#if HAVE_MERGE_GROUP
        // Optionally, add the newly-created vector grids to a group.
        if (!primsToGroup.empty() && mergeGroupStr.isstring()) {
            GA_PrimitiveGroup* mergeGroup =
                gdp->findPrimitiveGroup(mergeGroupStr.buffer());
            if (mergeGroup == NULL) {
                mergeGroup = gdp->newPrimitiveGroup(mergeGroupStr.buffer());
            }
            if (mergeGroup != NULL) {
                for (PrimVDBVec::iterator i = primsToGroup.begin(), e = primsToGroup.end();
                    i != e; ++i)
                {
                    mergeGroup->add(*i);
                }
            }
        }
#endif

        if (removeSourceGrids) {
            // Remove scalar grids that were merged.
            primsToRemove.erase(NULL);
            for (PrimVDBSet::iterator i = primsToRemove.begin(), e = primsToRemove.end();
                i != e; ++i)
            {
                gdp->destroyPrimitive(*(*i), /*andPoints=*/true);
            }
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
