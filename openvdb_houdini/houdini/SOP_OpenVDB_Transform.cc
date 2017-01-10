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
/// @file SOP_OpenVDB_Transform.cc
///
/// @author FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/VectorTransformer.h> // for transformVectors()
#include <UT/UT_Interrupt.h>
#include <boost/math/constants/constants.hpp>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Transform: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Transform(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Transform() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Group pattern
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to be transformed.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    // Translation
    parms.add(hutil::ParmFactory(PRM_XYZ_J, "t", "Translate")
        .setVectorSize(3)
        .setDefault(PRMzeroDefaults));

    // Rotation
    parms.add(hutil::ParmFactory(PRM_XYZ_J, "r", "Rotate")
        .setHelpText("Rotation specified in ZYX order")
        .setVectorSize(3)
        .setDefault(PRMzeroDefaults));

    // Scale
    parms.add(hutil::ParmFactory(PRM_XYZ_J, "s", "Scale")
        .setVectorSize(3)
        .setDefault(PRMoneDefaults));

    // Pivot
    parms.add(hutil::ParmFactory(PRM_XYZ_J, "p", "Pivot")
        .setVectorSize(3)
        .setDefault(PRMzeroDefaults));

    // Uniform scale
    parms.add(hutil::ParmFactory(PRM_FLT_J, "uniformScale", "Uniform Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_FREE, 10));

    // Toggle, inverse
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "invert", "Invert Transformation")
        .setDefault(PRMzeroDefaults));

    // Toggle, apply transform to vector values
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "xformvectors", "Transform Vectors")
        .setDefault(PRMzeroDefaults)
        .setHelpText(
            "Apply the transform to the voxel values of vector-valued grids,\n"
            "in accordance with those grids' Vector Type attributes.\n"));


    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Transform", SOP_OpenVDB_Transform::factory, parms, *table)
        .addInput("Input with VDB grids to transform");
}


OP_Node*
SOP_OpenVDB_Transform::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Transform(net, name, op);
}


SOP_OpenVDB_Transform::SOP_OpenVDB_Transform(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


namespace {

// Functor for use with GEOvdbProcessTypedGridVec3() to apply a transform
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


OP_ERROR
SOP_OpenVDB_Transform::cookMySop(OP_Context& context)
{
    try {
        typedef openvdb::math::AffineMap AffineMap;
        typedef openvdb::math::Transform Transform;

        hutil::ScopedInputLock lock(*this, context);

        const fpreal time = context.getTime();

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSourceStealable(0, context);

        // Get UI parameters
        openvdb::Vec3R t(evalVec3R("t", time)), r(evalVec3R("r", time)),
            s(evalVec3R("s", time)), p(evalVec3R("p", time));

        s *= evalFloat("uniformScale", 0, time);

        int flagInverse = evalInt("invert", 0, time);

        const bool xformVec = evalInt("xformvectors", 0, time);

        // Get the group of grids to be transformed.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        UT_AutoInterrupt progress("Transform");

        // Build up the transform matrix from the UI parameters
        const double deg2rad = boost::math::constants::pi<double>() / 180.0;

        openvdb::Mat4R mat(openvdb::Mat4R::identity());
        mat.preTranslate(p);
        mat.preRotate(openvdb::math::X_AXIS, deg2rad*r[0]);
        mat.preRotate(openvdb::math::Y_AXIS, deg2rad*r[1]);
        mat.preRotate(openvdb::math::Z_AXIS, deg2rad*r[2]);
        mat.preScale(s);
        mat.preTranslate(-p);
        mat.preTranslate(t);

        if (flagInverse) mat = mat.inverse();

        const VecXformOp xformOp(mat);

        // Construct an affine map.
        AffineMap map(mat);

        // For each VDB primitive in the given group...
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (progress.wasInterrupted()) throw std::runtime_error("Was Interrupted");

            GU_PrimVDB* vdb = *it;

            // No need to make the grid unique at this point, since we might not need
            // to modify its voxel data.
            hvdb::Grid& grid = vdb->getGrid();

            // Merge the transform's current affine representation with the new affine map.
            AffineMap::Ptr compound(
                new AffineMap(*grid.transform().baseMap()->getAffineMap(), map));

            // Simplify the affine map and replace the transform.
            grid.setTransform(Transform::Ptr(new Transform(openvdb::math::simplify(compound))));

            // Update the primitive's vertex position.
            /// @todo Need a simpler way to do this.
            hvdb::GridPtr copyOfGrid = grid.copyGrid();
            copyOfGrid->setTransform(grid.constTransform().copy());
            vdb->setGrid(*copyOfGrid);

            if (xformVec && vdb->getConstGrid().isInWorldSpace()
                && vdb->getConstGrid().getVectorType() != openvdb::VEC_INVARIANT)
            {
                // If (and only if) the grid is vector-valued, deep copy it,
                // then apply the transform to each voxel's value.
                GEOvdbProcessTypedGridVec3(*vdb, xformOp, /*makeUnique=*/true);
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
