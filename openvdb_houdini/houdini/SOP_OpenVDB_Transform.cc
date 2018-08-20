///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
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
#if UT_VERSION_INT >= 0x10050000 // 16.5.0 or later
#include <hboost/math/constants/constants.hpp>
#else
#include <boost/math/constants/constants.hpp>
#endif
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#if UT_MAJOR_VERSION_INT >= 16
#define VDB_COMPILABLE_SOP 1
#else
#define VDB_COMPILABLE_SOP 0
#endif

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;
#if UT_VERSION_INT < 0x10050000 // earlier than 16.5.0
namespace hboost = boost;
#endif


class SOP_OpenVDB_Transform: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Transform(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Transform() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

#if VDB_COMPILABLE_SOP
    class Cache: public SOP_VDBCacheOptions { OP_ERROR cookVDBSop(OP_Context&) override; };
#else
protected:
    OP_ERROR cookVDBSop(OP_Context&) override;
#endif
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of the input VDBs to be transformed.")
        .setDocumentation(
            "A subset of the input VDBs to be transformed"
            " (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_STRING, "xOrd", "Transform Order")
        .setDefault("tsr") ///< @todo Houdini default is "srt"
        .setChoiceList(&PRMtrsMenu)
        .setTypeExtended(PRM_TYPE_JOIN_PAIR)
        .setTooltip("The order in which transformations and rotations occur"));

    parms.add(hutil::ParmFactory(
        PRM_STRING | PRM_Type(PRM_Type::PRM_INTERFACE_LABEL_NONE), "rOrd", "")
        .setDefault("zyx") ///< @todo Houdini default is "xyz"
        .setChoiceList(&PRMxyzMenu));

    parms.add(hutil::ParmFactory(PRM_XYZ_J, "t", "Translate")
        .setVectorSize(3)
        .setDefault(PRMzeroDefaults)
        .setDocumentation("The amount of translation along the _x_, _y_ and _z_ axes"));

    parms.add(hutil::ParmFactory(PRM_XYZ_J, "r", "Rotate")
        .setVectorSize(3)
        .setDefault(PRMzeroDefaults)
        .setDocumentation("The amount of rotation about the _x_, _y_ and _z_ axes"));

    parms.add(hutil::ParmFactory(PRM_XYZ_J, "s", "Scale")
        .setVectorSize(3)
        .setDefault(PRMoneDefaults)
        .setDocumentation("Nonuniform scaling along the _x_, _y_ and _z_ axes"));

    parms.add(hutil::ParmFactory(PRM_XYZ_J, "p", "Pivot")
        .setVectorSize(3)
        .setDefault(PRMzeroDefaults)
        .setDocumentation("The pivot point for scaling and rotation"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "uniformScale", "Uniform Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_FREE, 10)
        .setDocumentation("Uniform scaling along all three axes"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "invert", "Invert Transformation")
        .setDefault(PRMzeroDefaults)
        .setDocumentation("Perform the inverse transformation."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "xformvectors", "Transform Vectors")
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "Apply the transform to the voxel values of vector-valued VDBs,\n"
            "in accordance with those VDBs' Vector Type attributes.\n")
        .setDocumentation(
            "Apply the transform to the voxel values of vector-valued VDBs,"
            " in accordance with those VDBs' __Vector Type__ attributes (as set,"
            " for example, with the [OpenVDB Create|Node:sop/DW_OpenVDBCreate] node)."));

    hvdb::OpenVDBOpFactory("OpenVDB Transform", SOP_OpenVDB_Transform::factory, parms, *table)
        .addInput("VDBs to transform")
#if VDB_COMPILABLE_SOP
        .setVerb(SOP_NodeVerb::COOK_INPLACE, []() { return new SOP_OpenVDB_Transform::Cache; })
#endif
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Modify the transforms of VDB volumes.\"\"\"\n\
\n\
@overview\n\
\n\
This node modifies the transform associated with each input VDB volume.\n\
It is usually preferable to use Houdini's native [Transform|Node:sop/xform] node,\n\
except if you want to also transform the _values_ of a vector-valued VDB.\n\
\n\
@related\n\
- [Node:sop/xform]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
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
VDB_NODE_OR_CACHE(VDB_COMPILABLE_SOP, SOP_OpenVDB_Transform)::cookVDBSop(OP_Context& context)
{
    try {
#if !VDB_COMPILABLE_SOP
        hutil::ScopedInputLock lock(*this, context);

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        lock.markInputUnlocked(0);
        duplicateSourceStealable(0, context);
#endif

        using AffineMap = openvdb::math::AffineMap;
        using Transform = openvdb::math::Transform;

        const fpreal time = context.getTime();

        // Get UI parameters
        openvdb::Vec3R t(evalVec3R("t", time)), r(evalVec3R("r", time)),
            s(evalVec3R("s", time)), p(evalVec3R("p", time));

        s *= evalFloat("uniformScale", 0, time);

        const auto xformOrder = evalStdString("xOrd", time);
        const auto rotOrder = evalStdString("rOrd", time);
        const bool flagInverse = evalInt("invert", 0, time);
        const bool xformVec = evalInt("xformvectors", 0, time);

        const auto isValidOrder = [](const std::string& expected, const std::string& actual) {
            if (actual.size() != expected.size()) return false;
            using CharSet = std::set<std::string::value_type>;
            return (CharSet(actual.begin(), actual.end())
                == CharSet(expected.begin(), expected.end()));
        };

        if (!isValidOrder("rst", xformOrder)) {
            std::ostringstream mesg;
            mesg << "Invalid transform order \"" << xformOrder
                << "\"; expected \"tsr\", \"rst\", etc.";
            throw std::runtime_error(mesg.str());
        }

        if (!isValidOrder("xyz", rotOrder)) {
            std::ostringstream mesg;
            mesg << "Invalid rotation order \"" << rotOrder
                << "\"; expected \"xyz\", \"zyx\", etc.";
            throw std::runtime_error(mesg.str());
        }

        // Get the group of grids to be transformed.
        const GA_PrimitiveGroup* group = matchGroup(*gdp, evalStdString("group", time));

        UT_AutoInterrupt progress("Transform");

        // Build up the transform matrix from the UI parameters
        const double deg2rad = hboost::math::constants::pi<double>() / 180.0;

        openvdb::Mat4R mat(openvdb::Mat4R::identity());
        const auto rotate = [&]() {
            for (auto axis = rotOrder.rbegin(); axis != rotOrder.rend(); ++axis) {
                switch (*axis) {
                    case 'x': mat.preRotate(openvdb::math::X_AXIS, deg2rad*r[0]); break;
                    case 'y': mat.preRotate(openvdb::math::Y_AXIS, deg2rad*r[1]); break;
                    case 'z': mat.preRotate(openvdb::math::Z_AXIS, deg2rad*r[2]); break;
                }
            }
        };
        if (xformOrder == "trs") {
            mat.preTranslate(p);
            mat.preScale(s);
            rotate();
            mat.preTranslate(-p);
            mat.preTranslate(t);
        } else if (xformOrder == "tsr") {
            mat.preTranslate(p);
            rotate();
            mat.preScale(s);
            mat.preTranslate(-p);
            mat.preTranslate(t);
        } else if (xformOrder == "rts") {
            mat.preTranslate(p);
            mat.preScale(s);
            mat.preTranslate(-p);
            mat.preTranslate(t);
            mat.preTranslate(p);
            rotate();
            mat.preTranslate(-p);
        } else if (xformOrder == "rst") {
            mat.preTranslate(t);
            mat.preTranslate(p);
            mat.preScale(s);
            rotate();
            mat.preTranslate(-p);
        } else if (xformOrder == "str") {
            mat.preTranslate(p);
            rotate();
            mat.preTranslate(-p);
            mat.preTranslate(t);
            mat.preTranslate(p);
            mat.preScale(s);
            mat.preTranslate(-p);
        } else /*if (xformOrder == "srt")*/ {
            mat.preTranslate(t);
            mat.preTranslate(p);
            rotate();
            mat.preScale(s);
            mat.preTranslate(-p);
        }

        if (flagInverse) mat = mat.inverse();

        const VecXformOp xformOp(mat);

        // Construct an affine map.
        AffineMap map(mat);

        // For each VDB primitive in the given group...
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (progress.wasInterrupted()) throw std::runtime_error("Interrupted");

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

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
