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
/// @file SOP_OpenVDB_Fill.cc
///
/// @author FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <UT/UT_Interrupt.h>

namespace hutil = houdini_utils;
namespace hvdb = openvdb_houdini;


class SOP_OpenVDB_Fill: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Fill(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Fill();

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to be processed.")
        .setChoiceList(&hutil::PrimGroupMenu));
    parms.add(hutil::ParmFactory(PRM_INT_XYZ, "min", "Min coord").setVectorSize(3));
    parms.add(hutil::ParmFactory(PRM_INT_XYZ, "max", "Max coord").setVectorSize(3));
    parms.add(hutil::ParmFactory(PRM_FLT_J, "value", "Value")
        .setTypeExtended(PRM_TYPE_JOIN_PAIR));
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "active", "Active")
        .setDefault(PRMoneDefaults));

    hvdb::OpenVDBOpFactory("OpenVDB Fill", SOP_OpenVDB_Fill::factory, parms, *table)
        .addInput("Input with VDB grids to operate on");
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

struct FillOp
{
    const openvdb::CoordBBox bbox;
    const float value;
    const bool active;

    FillOp(const openvdb::CoordBBox& b, float val, bool on):
        bbox(b), value(val), active(on)
    {}

    template<typename GridT>
    void operator()(GridT& grid) const
    {
        typedef typename GridT::ValueType ValueT;
        grid.fill(bbox, ValueT(openvdb::zeroVal<ValueT>() + value), active);
    }
};

} // unnamed namespace


OP_ERROR
SOP_OpenVDB_Fill::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal t = context.getTime();

        duplicateSource(0, context);

        UT_String groupStr;
        evalString(groupStr, "group", 0, t);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        const openvdb::CoordBBox bbox(
            openvdb::Coord(evalInt("min", 0, t), evalInt("min", 1, t), evalInt("min", 2, t)),
            openvdb::Coord(evalInt("max", 0, t), evalInt("max", 1, t), evalInt("max", 2, t)));

        const float value = evalFloat("value", 0, t);
        const bool active = evalInt("active", 0, t);

        const FillOp fillOp(bbox, value, active);

        UT_AutoInterrupt progress("Filling VDB grids");

        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (progress.wasInterrupted()) {
                throw std::runtime_error("processing was interrupted");
            }

            GU_PrimVDB* vdbPrim = *it;
            GEOvdbProcessTypedGrid(*vdbPrim, fillOp);
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
