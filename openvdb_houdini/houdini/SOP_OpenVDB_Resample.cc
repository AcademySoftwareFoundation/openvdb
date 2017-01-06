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
///
/// @file SOP_OpenVDB_Resample.cc
//
/// @author FX R&D OpenVDB team
///
/// @class SOP_OpenVDB_Resample
/// This node resamples voxels from input VDB grids into new grids
/// (of the same type) through a sampling transform that is either
/// specified by user-supplied translation, rotation, scale and pivot
/// parameters or taken from an optional reference grid.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/UT_VDBTools.h> // for GridTransformOp, et al.
#include <openvdb_houdini/UT_VDBUtils.h> // for UTvdbProcessTypedGridReal()
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <openvdb/tools/VectorTransformer.h> // for transformVectors()
#include <UT/UT_Interrupt.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Resample: public hvdb::SOP_NodeVDB
{
public:
    enum { MODE_PARMS = 0, MODE_REF_GRID, MODE_VOXEL_SIZE, MODE_VOXEL_SCALE };

    SOP_OpenVDB_Resample(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Resample() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i) const { return (i == 1); }

protected:
    struct RebuildOp;

    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
};


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Group pattern
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setHelpText("Specify a subset of the input\nVDB grids to be resampled"));

    // Reference grid group
    parms.add(hutil::ParmFactory(PRM_STRING, "reference_grid", "Reference")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setHelpText(
            "Specify a single reference grid from the\n"
            "first input whose transform is to be matched.\n"
            "Alternatively, connect the reference grid\n"
            "to the second input."));

    {
        const char* items[] = {
            "point",     "Nearest",
            "linear",    "Linear",
            "quadratic", "Quadratic",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "order", "Interpolation")
            .setDefault(PRMoneDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    {   // Transform source
        const char* items[] = {
            "explicit",       "Explicitly",
            "refvdb",         "To Match Reference VDB",
            "voxelsizeonly",  "Using Voxel Size Only",
            "voxelscaleonly", "Using Voxel Scale Only",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "mode", "Define Transform")
            .setDefault(PRMoneDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setHelpText(
                "Specify how to define the relative transform\n"
                "between an input and an output grid."));
    }

    parms.add(hutil::ParmFactory(PRM_ORD, "xOrd", "Transform Order")
        .setDefault(0, "tsr")
        .setChoiceList(&PRMtrsMenu)
        .setTypeExtended(PRM_TYPE_JOIN_PAIR));

    parms.add(hutil::ParmFactory(
        PRM_ORD | PRM_Type(PRM_Type::PRM_INTERFACE_LABEL_NONE), "rOrd", "")
        .setDefault(0, "zyx")
        .setChoiceList(&PRMxyzMenu));

    // Translation
    parms.add(hutil::ParmFactory(PRM_XYZ_J, "translate", "Translate")
        .setDefault(PRMzeroDefaults)
        .setVectorSize(3));

    // Rotation
    parms.add(hutil::ParmFactory(PRM_XYZ_J, "rotate", "Rotate")
        .setDefault(PRMzeroDefaults)
        .setVectorSize(3));

    // Scale
    parms.add(hutil::ParmFactory(PRM_XYZ_J, "scale", "Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.000001f, PRM_RANGE_UI, 10)
        .setVectorSize(3));

    // Pivot
    parms.add(hutil::ParmFactory(PRM_XYZ_J, "pivot", "Pivot")
        .setDefault(PRMzeroDefaults)
        .setVectorSize(3));

    // Voxel size
    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxel_size", "Voxel Size")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.000001f, PRM_RANGE_UI, 1)
        .setHelpText(
            "Specify the desired absolute voxel size for all output grids.\n"
            "Larger voxels correspond to lower resolution.\n"));

    // Voxel scale
    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelscale", "Voxel Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.000001f, PRM_RANGE_UI, 1)
        .setHelpText(
            "Specify the amount by which to scale the voxel size for each output grid.\n"
            "Larger voxels correspond to lower resolution.\n"));

    // Toggle to apply transform to vector values
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "xformvectors", "Transform Vectors")
        .setDefault(PRMzeroDefaults)
        .setHelpText(
            "Apply the resampling transform to the voxel values of\n"
            "vector-valued grids, in accordance with those grids'\n"
            "Vector Type attributes.\n"));

    // Level set rebuild toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "rebuild", "Rebuild SDF")
        .setDefault(PRMoneDefaults)
        .setHelpText(
            "Transforming (especially scaling) a level set might invalidate\n"
            "signed distances, necessitating reconstruction of the SDF.\n\n"
            "This option affects only level set grids, and it should\n"
            "almost always be enabled."));

    // Prune toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "prune", "Prune Tolerance")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText(
            "Collapse regions of constant value in output grids.\n"
            "Voxel values are considered equal if they differ\n"
            "by less than the specified threshold."));

    // Pruning tolerance slider
    parms.add(hutil::ParmFactory(
        PRM_FLT_J, "tolerance", "Prune Tolerance")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1));

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep2", "separator"));
    obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep3", "separator"));
    obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep4", "separator"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Resample", SOP_OpenVDB_Resample::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("Source VDB grids to resample")
        .addOptionalInput("Optional transform reference VDB grid");
}


// Disable UI Parms.
bool
SOP_OpenVDB_Resample::updateParmsFlags()
{
    bool changed = false;

    const int mode = evalInt("mode", 0, 0);
    changed |= enableParm("translate", mode == MODE_PARMS);
    changed |= enableParm("rotate", mode == MODE_PARMS);
    changed |= enableParm("scale", mode == MODE_PARMS);
    changed |= enableParm("pivot", mode == MODE_PARMS);
    changed |= enableParm("xOrd", mode == MODE_PARMS);
    changed |= enableParm("rOrd", mode == MODE_PARMS);
    changed |= enableParm("xformvectors", mode == MODE_PARMS);
    changed |= enableParm("reference_grid", mode == MODE_REF_GRID);
    changed |= enableParm("voxel_size", mode == MODE_VOXEL_SIZE);
    changed |= enableParm("voxelscale", mode == MODE_VOXEL_SCALE);
    // Show either the voxel size or the voxel scale parm, but not both.
    changed |= setVisibleState("voxel_size", mode != MODE_VOXEL_SCALE);
    changed |= setVisibleState("voxelscale", mode == MODE_VOXEL_SCALE);

    changed |= enableParm("tolerance", evalInt("prune", 0, 0));

    return changed;
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Resample::factory(OP_Network* net, const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Resample(net, name, op);
}


SOP_OpenVDB_Resample::SOP_OpenVDB_Resample(OP_Network* net, const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


// Helper class for use with UTvdbProcessTypedGrid()
struct SOP_OpenVDB_Resample::RebuildOp
{
    SOP_OpenVDB_Resample* self;
    openvdb::math::Transform xform;
    hvdb::GridPtr outGrid;

    template<typename GridT>
    void operator()(const GridT& grid)
    {
        typedef typename GridT::ValueType ValueT;

        const ValueT halfWidth = ValueT(grid.background() * (1.0 / grid.voxelSize()[0]));

        hvdb::Interrupter interrupter;
        try {
            outGrid = openvdb::tools::doLevelSetRebuild(grid,
                /*isovalue=*/openvdb::zeroVal<ValueT>(),
                /*exWidth=*/halfWidth, /*inWidth=*/halfWidth, &xform, &interrupter);
        } catch (openvdb::TypeError&) {
            self->addWarning(SOP_MESSAGE, ("skipped rebuild of level set grid "
                + grid.getName() + " of type " + grid.type()).c_str());
            outGrid = openvdb::ConstPtrCast<GridT>(grid.copy());
        }
    }
}; // struct RebuildOp


namespace {

// Functor for use with UTvdbProcessTypedGridVec3() to apply a transform
// to the voxel values of vector-valued grids
struct VecXformOp
{
    openvdb::Mat4d mat;
    VecXformOp(const openvdb::Mat4d& _mat): mat(_mat) {}
    template<typename GridT> void operator()(GridT& grid) const
    {
        openvdb::tools::transformVectors(grid, mat);
    }
};

} // unnamed namespace


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Resample::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal time = context.getTime();

        // This does a shallow copy of VDB grids and deep copy of native Houdini primitives.
        duplicateSource(0, context);

        const GU_Detail* refGdp = inputGeo(1, context);

        // Get parameters.
        const int samplingOrder = evalInt("order", 0, time);
        if (samplingOrder < 0 || samplingOrder > 2) {
            std::stringstream ss;
            ss << "expected interpolation order between 0 and 2, got "<< samplingOrder;
            throw std::runtime_error(ss.str().c_str());
        }

        UT_String xformOrder, rotOrder;
        evalString(xformOrder, "xOrd", 0, time);
        evalString(rotOrder, "rOrd", 0, time);

        const int mode = evalInt("mode", 0, time);
        if (mode < MODE_PARMS || mode > MODE_VOXEL_SCALE) {
            std::stringstream ss;
            ss << "expected mode between " << int(MODE_PARMS)
                << " and " << int(MODE_VOXEL_SCALE) << ", got "<< mode;
            throw std::runtime_error(ss.str().c_str());
        }

        const openvdb::Vec3R
            translate = SOP_NodeVDB::evalVec3R("translate", time),
            rotate = (M_PI / 180.0) * SOP_NodeVDB::evalVec3R("rotate", time),
            scale = SOP_NodeVDB::evalVec3R("scale", time),
            pivot = SOP_NodeVDB::evalVec3R("pivot", time);
        const float
            voxelSize = static_cast<float>(evalFloat("voxel_size", 0, time)),
            voxelScale = static_cast<float>(evalFloat("voxelscale", 0, time));

        const bool
            prune = evalInt("prune", 0, time),
            rebuild = evalInt("rebuild", 0, time),
            xformVec = evalInt("xformvectors", 0, time);
        const float tolerance = static_cast<float>(evalFloat("tolerance", 0, time));

        // Get the group of grids to be resampled.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        hvdb::GridCPtr refGrid;
        if (mode == MODE_VOXEL_SIZE) {
            // Create a dummy reference grid whose (linear) transform specifies
            // the desired voxel size.
            hvdb::GridPtr grid = openvdb::FloatGrid::create();
            grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));
            refGrid = grid;
        } else if (mode == MODE_VOXEL_SCALE) {
            // Create a dummy reference grid with a default (linear) transform.
            refGrid = openvdb::FloatGrid::create();
        } else if (mode == MODE_REF_GRID) {
            // Get the (optional) reference grid.
            UT_String refGroupStr;
            evalString(refGroupStr, "reference_grid", 0, time);
            const GA_PrimitiveGroup* refGroup = NULL;
            if (refGdp == NULL) {
                // If the second input is unconnected, the reference_grid parameter
                // specifies a reference grid from the first input.
                refGdp = gdp;
            }
            if (refGdp) {
                refGroup = matchGroup(*gdp, refGroupStr.toStdString());
                hvdb::VdbPrimCIterator it(refGdp, refGroup);
                if (it) {
                    refGrid = it->getConstGridPtr();
                    if (++it) {
                        addWarning(SOP_MESSAGE, "more than one reference grid was found");
                    }
                } else {
                    throw std::runtime_error("no reference grid was found");
                }
            }
        }

        UT_AutoInterrupt progress("Resampling VDB grids");

        // Iterate over the input grids.
        for (hvdb::VdbPrimIterator it(gdp, GA_Range::safedeletions(), group); it; ++it) {
            if (progress.wasInterrupted()) throw std::runtime_error("Was Interrupted");

            GU_PrimVDB* vdb = *it;

            const UT_VDBType valueType = vdb->getStorageType();

            const bool isLevelSet = ((vdb->getGrid().getGridClass() == openvdb::GRID_LEVEL_SET)
                && (valueType == UT_VDB_FLOAT || valueType == UT_VDB_DOUBLE));

            if (isLevelSet && !rebuild) {
                // If the input grid is a level set but level set rebuild is disabled,
                // set the grid's class to "unknown", to prevent the resample tool
                // from triggering a rebuild.
                vdb->getGrid().setGridClass(openvdb::GRID_UNKNOWN);
            }

            const hvdb::Grid& grid = vdb->getGrid();

            // Override the sampling order for boolean grids.
            int curOrder = samplingOrder;
            if (valueType == UT_VDB_BOOL && (samplingOrder != 0)) {
                std::stringstream ss;
                ss << "a boolean VDB grid can't be order-" << samplingOrder << " sampled;\n"
                    << "using nearest neighbor sampling instead";
                addWarning(SOP_MESSAGE, ss.str().c_str());
                curOrder = 0;
            }

            // Create a new, empty output grid of the same type as the input grid
            // and with the same metadata.
#ifdef OPENVDB_3_ABI_COMPATIBLE
            hvdb::GridPtr outGrid = grid.copyGrid(/*tree=*/openvdb::CP_NEW);
#else
            hvdb::GridPtr outGrid = grid.copyGridWithNewTree();
#endif

            UT_AutoInterrupt scopedInterrupt(
                ("Resampling " + it.getPrimitiveName().toStdString()).c_str());

            if (refGrid) {
                // If a reference grid was provided, then after resampling, the
                // output grid's transform will be the same as the reference grid's.

                openvdb::math::Transform::Ptr refXform = refGrid->transform().copy();
                if (mode == MODE_VOXEL_SCALE) {
                    openvdb::Vec3d scaledVoxelSize = grid.voxelSize() * voxelScale;
                    refXform->preScale(scaledVoxelSize);
                }

                if (isLevelSet && rebuild) {
                    // Use the level set rebuild tool to both resample and rebuild.
                    RebuildOp op;
                    op.self = this;
                    op.xform = *refXform;
                    UTvdbProcessTypedGridReal(valueType, grid, op);
                    outGrid = op.outGrid;

                } else {
                    // Use the resample tool to sample the input grid into the output grid.

                    // Set the correct transform on the output grid.
                    outGrid->setTransform(refXform);

                    if (curOrder == 0) {
                        hvdb::GridResampleToMatchOp<openvdb::tools::PointSampler> op(outGrid);
                        GEOvdbProcessTypedGridTopology(*vdb, op);
                    } else if (curOrder == 1) {
                        hvdb::GridResampleToMatchOp<openvdb::tools::BoxSampler> op(outGrid);
                        GEOvdbProcessTypedGridTopology(*vdb, op);
                    } else if (curOrder == 2) {
                        hvdb::GridResampleToMatchOp<openvdb::tools::QuadraticSampler> op(outGrid);
                        GEOvdbProcessTypedGridTopology(*vdb, op);
                    }
                }

            } else {
                // Resample into the output grid using the user-supplied transform.
                // The output grid's transform will be the same as the input grid's.

                openvdb::tools::GridTransformer xform(pivot, scale, rotate, translate,
                    xformOrder.toStdString(), rotOrder.toStdString());

                if (isLevelSet && rebuild) {
                    // Use the level set rebuild tool to both resample and rebuild.
                    RebuildOp op;
                    op.self = this;

                    // Compose the input grid's transform with the user-supplied transform.
                    // (The latter is retrieved from the GridTransformer, so that the
                    // order of operations and the rotation order are respected.)
                    op.xform = grid.constTransform();
                    op.xform.preMult(xform.getTransform().inverse());

                    UTvdbProcessTypedGridReal(valueType, grid, op);
                    outGrid = op.outGrid;
                    outGrid->setTransform(grid.constTransform().copy());

                } else {
                    // Use the resample tool to sample the input grid into the output grid.

                    hvdb::Interrupter interrupter;
                    xform.setInterrupter(interrupter);

                    if (curOrder == 0) {
                        hvdb::GridTransformOp<openvdb::tools::PointSampler> op(outGrid, xform);
                        GEOvdbProcessTypedGridTopology(*vdb, op);
                    } else if (curOrder == 1) {
                        hvdb::GridTransformOp<openvdb::tools::BoxSampler> op(outGrid, xform);
                        GEOvdbProcessTypedGridTopology(*vdb, op);
                    } else if (curOrder == 2) {
                        hvdb::GridTransformOp<openvdb::tools::QuadraticSampler> op(outGrid, xform);
                        GEOvdbProcessTypedGridTopology(*vdb, op);
                    }
                }

                if (xformVec && outGrid->isInWorldSpace()
                    && outGrid->getVectorType() != openvdb::VEC_INVARIANT)
                {
                    // If (and only if) the grid is vector-valued, apply the transform
                    // to each voxel's value.
                    VecXformOp op(xform.getTransform());
                    UTvdbProcessTypedGridVec3(valueType, *outGrid, op);
                }
            }

            if (prune) outGrid->pruneGrid(tolerance);

            // Replace the original VDB primitive with a new primitive that contains
            // the output grid and has the same attributes and group membership.
            hvdb::replaceVdbPrimitive(*gdp, outGrid, *vdb);
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
