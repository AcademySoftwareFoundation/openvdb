///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Fill.cc
///
/// @author FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <PRM/PRM_Parm.h>
#include <UT/UT_Interrupt.h>
#include <boost/utility/enable_if.hpp>
#include <boost/scoped_ptr.hpp>

namespace hutil = houdini_utils;
namespace hvdb = openvdb_houdini;


class SOP_OpenVDB_Fill: public hvdb::SOP_NodeVDB
{
public:
    enum Mode { MODE_INDEX = 0, MODE_WORLD, MODE_GEOM };

    SOP_OpenVDB_Fill(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Fill();

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned input) const { return (input == 1); }

protected:
    virtual bool updateParmsFlags();
    virtual void resolveObsoleteParms(PRM_ParmList*);
    virtual OP_ERROR cookMySop(OP_Context&);

    Mode getMode(fpreal time) const
    {
        UT_String modeStr;
        evalString(modeStr, "mode", 0, time);
        if (modeStr == "index") return MODE_INDEX;
        if (modeStr == "world") return MODE_WORLD;
        if (modeStr == "geom") return MODE_GEOM;

        std::string err = "unrecognized mode \"" + modeStr.toStdString() + "\"";
        throw std::runtime_error(err);
    }
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to be processed.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    {
        const char* items[] = {
            "index",  "Min and Max in Index Space",
            "world",  "Min and Max in World Space",
            "geom",   "Reference Geometry",
            NULL
        };
        parms.add(hutil::ParmFactory(PRM_STRING, "mode", "Bounds")
            .setDefault("index")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setHelpText(
                "Index Space:\n"
                "    Interpret the given min and max coordinates in index-space units.\n"
                "World Space:\n"
                "    Interpret the given min and max coordinates in world-space units.\n"
                "Reference Geometry:\n"
                "    Use the world-space bounds of the reference input geometry.\n"));
    }

    parms.add(hutil::ParmFactory(PRM_INT_XYZ, "min", "Min coord").setVectorSize(3));
    parms.add(hutil::ParmFactory(PRM_INT_XYZ, "max", "Max coord").setVectorSize(3));

    parms.add(hutil::ParmFactory(PRM_XYZ, "worldmin", "Min coord").setVectorSize(3));
    parms.add(hutil::ParmFactory(PRM_XYZ, "worldmax", "Max coord").setVectorSize(3));

    parms.add(hutil::ParmFactory(PRM_XYZ, "val", "Value").setVectorSize(3)
        .setTypeExtended(PRM_TYPE_JOIN_PAIR)
        .setHelpText(
            "The value with which to fill voxels\n"
            "(y and z are ignored when filling scalar grids)"));
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "active", "Active")
        .setDefault(PRMoneDefaults)
        .setHelpText(
            "If enabled, activate voxels in the fill region, otherwise deactivate them."));


    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "value", "Value"));


    hvdb::OpenVDBOpFactory("OpenVDB Fill", SOP_OpenVDB_Fill::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("Input with VDB grids to operate on")
        .addOptionalInput("Optional bounding geometry");
}


void
SOP_OpenVDB_Fill::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;

    PRM_Parm* parm = obsoleteParms->getParmPtr("value");
    if (parm && !parm->isFactoryDefault()) {
        // Transfer the scalar value of the obsolete parameter "value"
        // to the new, vector-valued parameter "val".
        const fpreal val = obsoleteParms->evalFloat("value", 0, /*time=*/0.0);
        setFloat("val", 0, 0.0, val);
        setFloat("val", 1, 0.0, val);
        setFloat("val", 2, 0.0, val);
    }

    // Delegate to the base class.
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}


bool
SOP_OpenVDB_Fill::updateParmsFlags()
{
    bool changed = false;
    const fpreal time = 0;

    //int refExists = (nInputs() == 2);

    Mode mode;
    try { mode = getMode(time); } catch (std::runtime_error&) { mode = MODE_INDEX; }

    switch (mode) {
        case MODE_INDEX:
            changed |= enableParm("min", true);
            changed |= enableParm("max", true);
            changed |= setVisibleState("min", true);
            changed |= setVisibleState("max", true);
            changed |= setVisibleState("worldmin", false);
            changed |= setVisibleState("worldmax", false);
            break;
        case MODE_WORLD:
            changed |= enableParm("worldmin", true);
            changed |= enableParm("worldmax", true);
            changed |= setVisibleState("min", false);
            changed |= setVisibleState("max", false);
            changed |= setVisibleState("worldmin", true);
            changed |= setVisibleState("worldmax", true);
            break;
        case MODE_GEOM:
            changed |= enableParm("min", false);
            changed |= enableParm("max", false);
            changed |= enableParm("worldmin", false);
            changed |= enableParm("worldmax", false);
            break;
    }

    return changed;
}


OP_Node*
SOP_OpenVDB_Fill::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Fill(net, name, op);
}


SOP_OpenVDB_Fill::SOP_OpenVDB_Fill(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


SOP_OpenVDB_Fill::~SOP_OpenVDB_Fill()
{
}


namespace {

// Convert a Vec3 value to a vector of another value type or to a scalar value

inline const openvdb::Vec3R& convertValue(const openvdb::Vec3R& val) { return val; }

// Overload for scalar types (discards all but the first vector component)
template<typename ValueType>
inline typename boost::disable_if_c<openvdb::VecTraits<ValueType>::IsVec, ValueType>::type
convertValue(const openvdb::Vec3R& val)
{
    return ValueType(val[0]);
}

// Overload for Vec2 types (not currently used)
template<typename ValueType>
inline typename boost::enable_if_c<
    openvdb::VecTraits<ValueType>::IsVec && openvdb::VecTraits<ValueType>::Size == 2,
    ValueType>::type
convertValue(const openvdb::Vec3R& val)
{
    typedef typename openvdb::VecTraits<ValueType>::ElementType ElemType;
    return ValueType(ElemType(val[0]), ElemType(val[1]));
}

// Overload for Vec3 types
template<typename ValueType>
inline typename boost::enable_if_c<
    openvdb::VecTraits<ValueType>::IsVec && openvdb::VecTraits<ValueType>::Size == 3,
    ValueType>::type
convertValue(const openvdb::Vec3R& val)
{
    typedef typename openvdb::VecTraits<ValueType>::ElementType ElemType;
    return ValueType(ElemType(val[0]), ElemType(val[1]), ElemType(val[2]));
}

// Overload for Vec4 types (not currently used)
template<typename ValueType>
inline typename boost::enable_if_c<
    openvdb::VecTraits<ValueType>::IsVec && openvdb::VecTraits<ValueType>::Size == 4,
    ValueType>::type
convertValue(const openvdb::Vec3R& val)
{
    typedef typename openvdb::VecTraits<ValueType>::ElementType ElemType;
    return ValueType(ElemType(val[0]), ElemType(val[1]), ElemType(val[2]), ElemType(1.0));
}


////////////////////////////////////////


struct FillOp
{
    const openvdb::CoordBBox indexBBox;
    const openvdb::BBoxd worldBBox;
    const openvdb::Vec3R value;
    const bool active;

    FillOp(const openvdb::CoordBBox& b, const openvdb::Vec3R& val, bool on):
        indexBBox(b), value(val), active(on)
    {}

    FillOp(const openvdb::BBoxd& b, const openvdb::Vec3R& val, bool on):
        worldBBox(b), value(val), active(on)
    {}

    template<typename GridT>
    void operator()(GridT& grid) const
    {
        openvdb::CoordBBox bbox = indexBBox;
        if (worldBBox) {
            openvdb::math::Vec3d imin, imax;
            openvdb::math::calculateBounds(grid.constTransform(),
               worldBBox.min(), worldBBox.max(), imin, imax);
            bbox.reset(openvdb::Coord::floor(imin), openvdb::Coord::ceil(imax));
        }
        typedef typename GridT::ValueType ValueT;
        grid.fill(bbox, convertValue<ValueT>(value), active);
    }
};

} // unnamed namespace


OP_ERROR
SOP_OpenVDB_Fill::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal t = context.getTime();

        duplicateSourceStealable(0, context);

        UT_String groupStr;
        evalString(groupStr, "group", 0, t);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        const openvdb::Vec3R value = SOP_NodeVDB::evalVec3R("val", t);
        const bool active = evalInt("active", 0, t);

        boost::scoped_ptr<const FillOp> fillOp;
        switch (getMode(t)) {
            case MODE_INDEX:
            {
                const openvdb::CoordBBox bbox(
                    openvdb::Coord(
                        evalInt("min", 0, t), evalInt("min", 1, t), evalInt("min", 2, t)),
                    openvdb::Coord(
                        evalInt("max", 0, t), evalInt("max", 1, t), evalInt("max", 2, t)));
                fillOp.reset(new FillOp(bbox, value, active));
                break;
            }
            case MODE_WORLD:
            {
                const openvdb::BBoxd bbox(
                    openvdb::BBoxd::ValueType(
                        evalFloat("worldmin", 0, t),
                        evalFloat("worldmin", 1, t),
                        evalFloat("worldmin", 2, t)),
                    openvdb::BBoxd::ValueType(
                        evalFloat("worldmax", 0, t),
                        evalFloat("worldmax", 1, t),
                        evalFloat("worldmax", 2, t)));
                fillOp.reset(new FillOp(bbox, value, active));
                break;
            }
            case MODE_GEOM:
            {
                openvdb::BBoxd bbox;
                if (const GU_Detail* refGeo = inputGeo(1)) {
                    UT_BoundingBox b;
                    refGeo->computeQuickBounds(b);
                    if (!b.isValid()) {
                        throw std::runtime_error("no reference geometry found");
                    }
                    bbox.min()[0] = b.xmin();
                    bbox.min()[1] = b.ymin();
                    bbox.min()[2] = b.zmin();
                    bbox.max()[0] = b.xmax();
                    bbox.max()[1] = b.ymax();
                    bbox.max()[2] = b.zmax();
                } else {
                    throw std::runtime_error("reference input is unconnected");
                }
                fillOp.reset(new FillOp(bbox, value, active));
                break;
            }
        }

        UT_AutoInterrupt progress("Filling VDB grids");

        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (progress.wasInterrupted()) {
                throw std::runtime_error("processing was interrupted");
            }

            GU_PrimVDB* vdbPrim = *it;
            GEOvdbProcessTypedGridTopology(*vdbPrim, *fillOp);
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
