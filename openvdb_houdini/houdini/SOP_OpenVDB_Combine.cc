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
/// @file SOP_OpenVDB_Combine.cc
///
/// @author FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/GridTransformer.h> // for resampleToMatch()
#include <openvdb/tools/LevelSetRebuild.h> // for levelSetRebuild()
#include <openvdb/tools/Morphology.h> // for deactivate()
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/util/NullInterrupter.h>
#include <PRM/PRM_Parm.h>
#include <UT/UT_Interrupt.h>
#include <boost/math/special_functions/fpclassify.hpp> // for isfinite()
#include <sstream>


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


/// @brief SOP to combine two VDB grids via various arithmetic operations
class SOP_OpenVDB_Combine: public hvdb::SOP_NodeVDB
{
public:
    enum Operation {
        OP_COPY_A,            // A
        OP_COPY_B,            // B
        OP_INVERT,            // 1 - A
        OP_ADD,               // A + B
        OP_SUBTRACT,          // A - B
        OP_MULTIPLY,          // A * B
        OP_DIVIDE,            // A / B
        OP_MAXIMUM,           // max(A, B)
        OP_MINIMUM,           // min(A, B)
        OP_BLEND1,            // (1 - A) * B
        OP_BLEND2,            // A + (1 - A) * B
        OP_UNION,             // CSG A u B
        OP_INTERSECTION,      // CSG A n B
        OP_DIFFERENCE,        // CSG A / B
        OP_REPLACE,           // replace A with B
        OP_TOPO_UNION,        // A u active(B)
        OP_TOPO_INTERSECTION, // A n active(B)
        OP_TOPO_DIFFERENCE    // A / active(B)
    };
    enum { OP_FIRST = OP_COPY_A, OP_LAST = OP_TOPO_DIFFERENCE };

    static const char* const sOpMenuItems[];

    static Operation asOp(int i, Operation defaultOp = OP_COPY_A)
    {
        return (i >= OP_FIRST && i <= OP_LAST)
            ? static_cast<Operation>(i) : defaultOp;
    }

    enum ResampleMode {
        RESAMPLE_OFF,    // don't auto-resample grids
        RESAMPLE_B,      // resample B to match A
        RESAMPLE_A,      // resample A to match B
        RESAMPLE_HI_RES, // resample higher-res grid to match lower-res
        RESAMPLE_LO_RES  // resample lower-res grid to match higher-res
    };
    enum { RESAMPLE_MODE_FIRST = RESAMPLE_OFF, RESAMPLE_MODE_LAST = RESAMPLE_LO_RES };

    static const char* const sResampleModeMenuItems[];

    static ResampleMode asResampleMode(int i, ResampleMode defaultMode = RESAMPLE_B)
    {
        return (i >= RESAMPLE_MODE_FIRST && i <= RESAMPLE_MODE_LAST)
            ? static_cast<ResampleMode>(i) : defaultMode;
    }

    SOP_OpenVDB_Combine(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Combine() {}

    static OP_Node* factory(OP_Network*, const char*, OP_Operator*);

    fpreal getTime() { return mTime; }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
    virtual void resolveObsoleteParms(PRM_ParmList*);

private:
    fpreal mTime;
    bool mWasCompositeSOP;

    template<typename> struct DispatchOp;
    struct CombineOp;

    hvdb::GridPtr combineGrids(Operation,
        hvdb::GridCPtr aGrid, hvdb::GridCPtr bGrid,
        const UT_String& aGridName, const UT_String& bGridName,
        ResampleMode resample);

    bool needAGrid(Operation op) const
        { return (op != OP_COPY_B); }
    bool needBGrid(Operation op) const
        { return (op != OP_COPY_A && op != OP_INVERT); }
    bool needLevelSets(Operation op) const
        { return (op == OP_UNION || op == OP_INTERSECTION || op == OP_DIFFERENCE); }
};


//#define TIMES " \xd7 " // ISO-8859 multiplication symbol
#define TIMES " * "
const char* const SOP_OpenVDB_Combine::sOpMenuItems[] = {
    "copya",                "Copy A",
    "copyb",                "Copy B",
    "inverta",              "Invert A",
    "add",                  "Add",
    "subtract",             "Subtract",
    "multiply",             "Multiply",
    "divide",               "Divide",
    "maximum",              "Maximum",
    "minimum",              "Minimum",
    "compatimesb",          "(1 - A)" TIMES "B",
    "apluscompatimesb",     "A + (1 - A)" TIMES "B",
    "sdfunion",             "SDF Union",
    "sdfintersect",         "SDF Intersection",
    "sdfdifference",        "SDF Difference",
    "replacewithactive",    "Replace A with Active B",
    "topounion",            "Activity Union",
    "topointersect",        "Activity Intersection",
    "topodifference",       "Activity Difference",
    NULL
};
#undef TIMES

const char* const SOP_OpenVDB_Combine::sResampleModeMenuItems[] = {
    "off",      "Off",
    "btoa",     "B to Match A",
    "atob",     "A to Match B",
    "hitolo",   "Higher-res to Match Lower-res",
    "lotohi",   "Lower-res to Match Higher-res",
    NULL
};


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Group A
    parms.add(hutil::ParmFactory(PRM_STRING, "groupA", "Group A")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setHelpText("Use a subset of the first input as the A grid(s)."));

    // Group B
    parms.add(hutil::ParmFactory(PRM_STRING, "groupB", "Group B")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setHelpText("Use a subset of the second input as the B grid(s)."));

    // Toggle to enable flattening B into A.
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "flatten", "Flatten All B into A")
        .setDefault(PRMzeroDefaults));

    // Toggle to enable/disable A/B pairing
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "pairs", "Combine A/B Pairs")
        .setDefault(PRMoneDefaults)
        .setHelpText(
            "If disabled, combine each grid in group A\n"
            "with the first grid in group B.  Otherwise,\n"
            "pair A and B grids in the order that they\n"
            "appear in their respective groups."));


    // Menu of available operations
    parms.add(hutil::ParmFactory(PRM_ORD, "operation", "Operation")
        .setDefault(PRMzeroDefaults)
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, SOP_OpenVDB_Combine::sOpMenuItems));

    // Scalar multiplier on the A grid
    parms.add(hutil::ParmFactory(PRM_FLT_J, "mult_a", "A Multiplier")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_UI, -10, PRM_RANGE_UI, 10)
        .setHelpText(
            "Multiply voxel values in the A grid by a scalar\n"
            "before combining the A grid with the B grid."));

    // Scalar multiplier on the B grid
    parms.add(hutil::ParmFactory(PRM_FLT_J, "mult_b", "B Multiplier")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_UI, -10, PRM_RANGE_UI, 10)
        .setHelpText(
            "Multiply voxel values in the B grid by a scalar\n"
            "before combining the A grid with the B grid."));

    // Menu of resampling options
    parms.add(hutil::ParmFactory(PRM_ORD, "resample", "Resample")
        .setDefault(PRMoneDefaults)
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, SOP_OpenVDB_Combine::sResampleModeMenuItems)
        .setHelpText(
            "If the A and B grids have different transforms, one grid should\n"
            "be resampled to match the other before the two are combined.\n"
            "Also, level set grids should have matching background values.\n"));
    {
        // Menu of resampling interpolation order options
        const char* items[] = {
            "point",     "Nearest",
            "linear",    "Linear",
            "quadratic", "Quadratic",
            NULL
        };
        parms.add(hutil::ParmFactory(PRM_ORD, "resampleinterp", "Interpolation")
            .setDefault(PRMoneDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setHelpText(
                "Specify the type of interpolation to be used when\n"
                "resampling one grid to match the other's transform."));
    }

    // Deactivate background value toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "deactivate", "Deactivate Background Voxels")
        .setDefault(PRMzeroDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));

    // Deactivation tolerance slider
    parms.add(hutil::ParmFactory(PRM_FLT_J, "bgtolerance", "Deactivate Tolerance")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1)
        .setHelpText(
            "Deactivate active output voxels whose values\n"
            "equal the output grid's background value.\n"
            "Voxel values are considered equal if they differ\n"
            "by less than the specified tolerance."));

    // Prune toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "prune", "Prune")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));

    // Pruning tolerance slider
    parms.add(hutil::ParmFactory(PRM_FLT_J, "tolerance", "Prune Tolerance")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1)
        .setHelpText(
            "Collapse regions of constant value in output grids.\n"
            "Voxel values are considered equal if they differ\n"
            "by less than the specified tolerance."));

    // Flood fill toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "flood", "Signed-Flood-Fill Output SDFs")
        .setDefault(PRMzeroDefaults)
        .setHelpText(
            "Reclassify inactive voxels of level set grids as either inside or outside."));

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "combination", "Operation")
        .setDefault(-2));
    obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep1", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep2", ""));


    // Register SOP
    hvdb::OpenVDBOpFactory("OpenVDB Combine", SOP_OpenVDB_Combine::factory, parms, *table)
        .addAlias("OpenVDB Composite")
        .addAlias("OpenVDB CSG")
        .setObsoleteParms(obsoleteParms)
        .addInput("A VDBs")
        .addOptionalInput("B VDBs");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Combine::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Combine(net, name, op);
}


SOP_OpenVDB_Combine::SOP_OpenVDB_Combine(
    OP_Network* net, const char* name, OP_Operator* op)
    : SOP_NodeVDB(net, name, op)
    , mTime(0.0)
    , mWasCompositeSOP(UT_String(name).fcontain("Composite"))
        // if this SOP's name contains "Composite", assume that it was formerly
        // a DW_OpenVDBComposite SOP and not a DW_OpenVDBCSG SOP
{
}


////////////////////////////////////////


void
SOP_OpenVDB_Combine::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;

    PRM_Parm* parm = obsoleteParms->getParmPtr("combination");
    if (parm && (!parm->isFactoryDefault() || !mWasCompositeSOP)) {
        // The "combination" choices (union, intersection, difference) from
        // the old CSG SOP were appended to this SOP's "operation" list.
        switch (obsoleteParms->evalInt("combination", 0, /*time=*/0.0)) {
            case 0: setInt("operation", 0, 0.0, OP_UNION); break;
            case 1: setInt("operation", 0, 0.0, OP_INTERSECTION); break;
            case 2: setInt("operation", 0, 0.0, OP_DIFFERENCE); break;
        }
    }

    // Delegate to the base class.
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}


////////////////////////////////////////

// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_Combine::updateParmsFlags()
{
    bool changed = false;

    changed |= enableParm("resampleinterp", evalInt("resample", 0, 0) != 0);
    changed |= enableParm("bgtolerance", evalInt("deactivate", 0, 0) != 0);
    changed |= enableParm("tolerance", evalInt("prune", 0, 0) != 0);
    changed |= enableParm("pairs", evalInt("flatten", 0, 0) == 0);

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Combine::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        duplicateSource(0, context);

        mTime = context.getTime();

        const bool pairs = evalInt("pairs", /*idx=*/0, getTime());
        const bool flatten = evalInt("flatten", /*idx=*/0, getTime());
        const Operation op = asOp(evalInt("operation", 0, getTime()));
        const bool needA = needAGrid(op), needB = needBGrid(op);
        const ResampleMode resample = asResampleMode(evalInt("resample", 0, getTime()));

        GU_Detail* aGdp = gdp;
        const GU_Detail* bGdp = inputGeo(1, context);

        UT_String aGroupStr, bGroupStr;
        evalString(aGroupStr, "groupA", 0, getTime());
        evalString(bGroupStr, "groupB", 0, getTime());

        const GA_PrimitiveGroup
            *aGroup = matchGroup(*aGdp, aGroupStr.toStdString()),
            *bGroup = (!bGdp ? NULL : matchGroup(const_cast<GU_Detail&>(*bGdp),
                bGroupStr.toStdString()));

        UT_AutoInterrupt progress("Combining VDB grids");

        // Iterate over A and, optionally, B grids.
        hvdb::VdbPrimIterator aIt(aGdp, GA_Range::safedeletions(), aGroup);
        hvdb::VdbPrimCIterator bIt(bGdp, bGroup);
        for ( ; (!needA || aIt) && (!needB || bIt); ++aIt, ((needB && pairs) ? ++bIt : bIt))
        {
            if (progress.wasInterrupted()) {
                throw std::runtime_error("was interrupted");
            }

            // Note: even if needA is false, we still need to delete A grids.
            GU_PrimVDB* aVdb = aIt ? *aIt : NULL;

            const GU_PrimVDB* bVdb = bIt ? *bIt : NULL;
            hvdb::GridPtr aGrid;
            hvdb::GridCPtr bGrid;

            if (aVdb) aGrid = aVdb->getGridPtr();
            if (bVdb) bGrid = bVdb->getConstGridPtr();

            // For error reporting, get the names of the A and B grids.
            UT_String aGridName = aIt.getPrimitiveName(/*default=*/"A");
            UT_String bGridName = bIt.getPrimitiveName(/*default=*/"B");

            // Name the output grid after the A grid, except (see below) if the A grid is unused.
            UT_String outGridName = aIt.getPrimitiveName();

            hvdb::GridPtr outGrid;

            while (true) {
                // If the A grid is unused, name the output grid after the most recent B grid.
                if (!needA) outGridName = bIt.getPrimitiveName();

                outGrid = combineGrids(op, aGrid, bGrid, aGridName, bGridName, resample);

                // When not flattening, quit after one pass.
                if (!flatten) break;

                // See if we have any more B grids.
                ++bIt;
                if (!bIt) break;

                bVdb = *bIt;
                bGrid = bVdb->getConstGridPtr();
                bGridName = bIt.getPrimitiveName(/*default=*/"B");

                aGrid = outGrid;
                if (!aGrid) break;
            }

            if (outGrid) {
                // Add a new VDB primitive for the output grid to the output gdp.
                GU_PrimVDB::buildFromGrid(*gdp, outGrid,
                    /*copyAttrsFrom=*/needA ? aVdb : bVdb, outGridName);

                // Remove the A grid from the output gdp.
                if (aVdb) gdp->destroyPrimitive(*aVdb, /*andPoints=*/true);
            }

            if (!needA && !pairs) break;
            if (flatten) break;
        }

        // In non-paired mode, there should be only one B grid.
        if (!pairs && !flatten) ++bIt;

        // In flatten mode there should be a single A grid.
        if (flatten) ++aIt;

        const bool unusedA = (needA && aIt), unusedB = (needB && bIt);
        if (unusedA || unusedB) {
            std::ostringstream ostr;
            ostr << "some grids were not processed because there were more "
                << (unusedA ? "A" : "B") << " grids than "
                << (unusedA ? "B" : "A") << " grids";
            addWarning(SOP_MESSAGE, ostr.str().c_str());
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}


////////////////////////////////////////


namespace {

/// Functor to compute scale * grid + offset, for scalars scale and offset
template<typename GridT>
struct MulAdd
{
    typedef typename GridT::ValueType ValueT;
    typedef typename GridT::Ptr GridPtrT;

    float scale, offset;

    explicit MulAdd(float s, float t = 0.0): scale(s), offset(t) {}

    void operator()(const ValueT& a, const ValueT&, ValueT& out) const
        { out = ValueT(a * scale + offset); }

    /// @return true if the scale is 1 and the offset is 0
    bool isIdentity() const
    {
        return (openvdb::math::isApproxEqual(scale, 1.f, 1.0e-6f)
            && openvdb::math::isApproxEqual(offset, 0.f, 1.0e-6f));
    }

    /// Compute dest = src * scale + offset
    void process(const GridT& src, GridPtrT& dest) const
    {
        if (isIdentity()) {
            dest = src.deepCopy();
        } else {
            if (!dest) dest = GridT::create(src); // same transform, new tree
            ValueT bg;
            (*this)(src.background(), ValueT(), bg);
            openvdb::tools::changeBackground(dest->tree(), bg);
            dest->tree().combine2(src.tree(), src.tree(), *this, /*prune=*/false);
        }
    }
};


////////////////////////////////////////


/// Functor to compute (1 - A) * B for grids A and B
template<typename ValueT>
struct Blend1
{
    float aMult, bMult;
    const ValueT ONE;
    explicit Blend1(float a = 1.0, float b = 1.0):
        aMult(a), bMult(b), ONE(openvdb::zeroVal<ValueT>() + 1) {}
    void operator()(const ValueT& a, const ValueT& b, ValueT& out) const
        { out = ValueT((ONE - aMult * a) * bMult * b); }
};


////////////////////////////////////////


/// Functor to compute A + (1 - A) * B for grids A and B
template<typename ValueT>
struct Blend2
{
    float aMult, bMult;
    const ValueT ONE;
    explicit Blend2(float a = 1.0, float b = 1.0):
        aMult(a), bMult(b), ONE(openvdb::zeroVal<ValueT>() + 1) {}
    void operator()(const ValueT& a, const ValueT& b, ValueT& out) const
        { out = ValueT(a*aMult); out = out + ValueT((ONE - out) * bMult*b); }
};


////////////////////////////////////////


// Helper class to compare both scalar and vector values
template<typename ValueT>
struct ApproxEq
{
    const ValueT &a, &b;
    ApproxEq(const ValueT& _a, const ValueT& _b): a(_a), b(_b) {}
    operator bool() const {
        return openvdb::math::isRelOrApproxEqual(
            a, b, /*rel*/ValueT(1e-6f), /*abs*/ValueT(1e-8f));
    }
};


// Specialization for Vec2
template<typename T>
struct ApproxEq<openvdb::math::Vec2<T> >
{
    typedef openvdb::math::Vec2<T> VecT;
    typedef typename VecT::value_type ValueT;
    const VecT &a, &b;
    ApproxEq(const VecT& _a, const VecT& _b): a(_a), b(_b) {}
    operator bool() const { return a.eq(b, /*abs=*/ValueT(1e-8f)); }
};


// Specialization for Vec3
template<typename T>
struct ApproxEq<openvdb::math::Vec3<T> >
{
    typedef openvdb::math::Vec3<T> VecT;
    typedef typename VecT::value_type ValueT;
    const VecT &a, &b;
    ApproxEq(const VecT& _a, const VecT& _b): a(_a), b(_b) {}
    operator bool() const { return a.eq(b, /*abs=*/ValueT(1e-8f)); }
};


// Specialization for Vec4
template<typename T>
struct ApproxEq<openvdb::math::Vec4<T> >
{
    typedef openvdb::math::Vec4<T> VecT;
    typedef typename VecT::value_type ValueT;
    const VecT &a, &b;
    ApproxEq(const VecT& _a, const VecT& _b): a(_a), b(_b) {}
    operator bool() const { return a.eq(b, /*abs=*/ValueT(1e-8f)); }
};


template<typename T> inline bool isFinite(const T& val) { return boost::math::isfinite(val); }
inline bool isFinite(bool) { return true; }

} // unnamed namespace


////////////////////////////////////////


template<typename AGridT>
struct SOP_OpenVDB_Combine::DispatchOp
{
    SOP_OpenVDB_Combine::CombineOp* combineOp;

    DispatchOp(SOP_OpenVDB_Combine::CombineOp& op): combineOp(&op) {}

    template<typename BGridT> void operator()(typename BGridT::ConstPtr);
}; // struct DispatchOp


// Helper class for use with UTvdbProcessTypedGrid()
struct SOP_OpenVDB_Combine::CombineOp
{
    SOP_OpenVDB_Combine* self;
    Operation op;
    ResampleMode resample;
    UT_String aGridName, bGridName;
    hvdb::GridCPtr aBaseGrid, bBaseGrid;
    hvdb::GridPtr outGrid;
    hvdb::Interrupter interrupt;

    CombineOp(): self(NULL) {}

    // Functor for use with UTvdbProcessTypedGridScalar() to return
    // a scalar grid's background value as a floating-point quantity
    struct BackgroundOp {
        double value;
        BackgroundOp(): value(0.0) {}
        template<typename GridT> void operator()(const GridT& grid) {
            value = static_cast<double>(grid.background());
        }
    };
    static double getScalarBackgroundValue(const hvdb::Grid& baseGrid)
    {
        BackgroundOp bgOp;
        UTvdbProcessTypedGridScalar(UTvdbGetGridType(baseGrid), baseGrid, bgOp);
        return bgOp.value;
    }

    template<typename GridT>
    typename GridT::Ptr resampleToMatch(const GridT& src, const hvdb::Grid& ref, int order)
    {
        typedef typename GridT::ValueType ValueT;
        const ValueT ZERO = openvdb::zeroVal<ValueT>();

        const openvdb::math::Transform& refXform = ref.constTransform();

        typename GridT::Ptr dest;
        if (src.getGridClass() == openvdb::GRID_LEVEL_SET) {
            // For level set grids, use the level set rebuild tool to both resample the
            // source grid to match the reference grid and to rebuild the resulting level set.
            const ValueT halfWidth = ((ref.getGridClass() == openvdb::GRID_LEVEL_SET)
                ? ValueT(ZERO + this->getScalarBackgroundValue(ref) * (1.0 / ref.voxelSize()[0]))
                : ValueT(src.background() * (1.0 / src.voxelSize()[0])));

            if (!isFinite(halfWidth)) {
                std::stringstream msg;
                msg << "Resample to match: Illegal narrow band width = " << halfWidth
                    << ", caused by grid '" << src.getName() << "' with background "
                    << this->getScalarBackgroundValue(ref);
                throw std::invalid_argument(msg.str());
            }

            try {
                dest = openvdb::tools::doLevelSetRebuild(src, /*iso=*/ZERO,
                    /*exWidth=*/halfWidth, /*inWidth=*/halfWidth, &refXform, &interrupt);
            } catch (openvdb::TypeError&) {
                self->addWarning(SOP_MESSAGE, ("skipped rebuild of level set grid "
                    + src.getName() + " of type " + src.type()).c_str());
                dest.reset();
            }
        }
        if (!dest && src.constTransform() != refXform) {
            // For non-level set grids or if level set rebuild failed due to an unsupported
            // grid type, use the grid transformer tool to resample the source grid to match
            // the reference grid.
#ifdef OPENVDB_3_ABI_COMPATIBLE
            dest = src.copy(openvdb::CP_NEW);
#else
            dest = src.copyWithNewTree();
#endif
            dest->setTransform(refXform.copy());
            using namespace openvdb;
            switch (order) {
            case 0: tools::resampleToMatch<tools::PointSampler>(src, *dest, interrupt); break;
            case 1: tools::resampleToMatch<tools::BoxSampler>(src, *dest, interrupt); break;
            case 2: tools::resampleToMatch<tools::QuadraticSampler>(src, *dest, interrupt); break;
            }
        }
        return dest;
    }

    // If necessary, resample one grid so that its index space registers
    // with the other grid's.
    // Note that one of the grid pointers might change as a result.
    template<typename AGridT, typename BGridT>
    void resampleGrids(const AGridT*& aGrid, const BGridT*& bGrid)
    {
        if (!aGrid || !bGrid) return;

        const bool
            needA = self->needAGrid(op),
            needB = self->needBGrid(op),
            needBoth = needA && needB;
        const int samplingOrder = self->evalInt("resampleinterp", 0, self->getTime());

        // One of RESAMPLE_A, RESAMPLE_B or RESAMPLE_OFF, specifying whether
        // grid A, grid B or neither grid was resampled
        int resampleWhich = RESAMPLE_OFF;

        // Determine which of the two grids should be resampled.
        if (resample == RESAMPLE_HI_RES || resample == RESAMPLE_LO_RES) {
            const openvdb::Vec3d
                aVoxSize = aGrid->voxelSize(),
                bVoxSize = bGrid->voxelSize();
            const double
                aVoxVol = aVoxSize[0] * aVoxSize[1] * aVoxSize[2],
                bVoxVol = bVoxSize[0] * bVoxSize[1] * bVoxSize[2];
            resampleWhich = ((aVoxVol > bVoxVol && resample == RESAMPLE_LO_RES)
                || (aVoxVol < bVoxVol && resample == RESAMPLE_HI_RES))
                ? RESAMPLE_A : RESAMPLE_B;
        } else {
            resampleWhich = resample;
        }

        if (aGrid->constTransform() != bGrid->constTransform()) {
            // If the A and B grid transforms don't match, one of the grids
            // should be resampled into the other's index space.
            if (resample == RESAMPLE_OFF) {
                if (needBoth) {
                    // Resampling is disabled.  Just log a warning.
                    std::ostringstream ostr;
                    ostr << aGridName << " and " << bGridName << " transforms don't match";
                    self->addWarning(SOP_MESSAGE, ostr.str().c_str());
                }
            } else {
                if (needA && resampleWhich == RESAMPLE_A) {
                    // Resample grid A into grid B's index space.
                    aBaseGrid = this->resampleToMatch(*aGrid, *bGrid, samplingOrder);
                    aGrid = static_cast<const AGridT*>(aBaseGrid.get());
                } else if (needB && resampleWhich == RESAMPLE_B) {
                    // Resample grid B into grid A's index space.
                    bBaseGrid = this->resampleToMatch(*bGrid, *aGrid, samplingOrder);
                    bGrid = static_cast<const BGridT*>(bBaseGrid.get());
                }
            }
        }

        if (aGrid->getGridClass() == openvdb::GRID_LEVEL_SET &&
            bGrid->getGridClass() == openvdb::GRID_LEVEL_SET)
        {
            // If both grids are level sets, ensure that their background values match.
            // (If one of the grids was resampled, then the background values should
            // already match.)
            const double
                a = this->getScalarBackgroundValue(*aGrid),
                b = this->getScalarBackgroundValue(*bGrid);
            if (!ApproxEq<double>(a, b)) {
                if (resample == RESAMPLE_OFF) {
                    if (needBoth) {
                        // Resampling/rebuilding is disabled.  Just log a warning.
                        std::ostringstream ostr;
                        ostr << aGridName << " and " << bGridName
                            << " background values don't match ("
                            << std::setprecision(3) << a << " vs. " << b << ");\n"
                            << "                 the output grid will not be a valid level set";
                        self->addWarning(SOP_MESSAGE, ostr.str().c_str());
                    }
                } else {
                    // One of the two grids needs a level set rebuild.
                    if (needA && resampleWhich == RESAMPLE_A) {
                        // Rebuild A to match B's background value.
                        aBaseGrid = this->resampleToMatch(*aGrid, *bGrid, samplingOrder);
                        aGrid = static_cast<const AGridT*>(aBaseGrid.get());
                    } else if (needB && resampleWhich == RESAMPLE_B) {
                        // Rebuild B to match A's background value.
                        bBaseGrid = this->resampleToMatch(*bGrid, *aGrid, samplingOrder);
                        bGrid = static_cast<const BGridT*>(bBaseGrid.get());
                    }
                }
            }
        }
    }

    void checkVectorTypes(const hvdb::Grid* aGrid, const hvdb::Grid* bGrid)
    {
        if (!aGrid || !bGrid || !self->needAGrid(op) || !self->needBGrid(op)) return;

        switch (op) {
            case OP_TOPO_UNION:
            case OP_TOPO_INTERSECTION:
            case OP_TOPO_DIFFERENCE:
                // No need to warn about different vector types for topology-only operations.
                break;

            default:
            {
                const openvdb::VecType
                    aVecType = aGrid->getVectorType(),
                    bVecType = bGrid->getVectorType();
                if (aVecType != bVecType) {
                    std::ostringstream ostr;
                    ostr << aGridName << " and " << bGridName
                        << " have different vector types\n"
                        << "                 (" << hvdb::Grid::vecTypeToString(aVecType)
                        << " vs. " << hvdb::Grid::vecTypeToString(bVecType) << ")";
                    self->addWarning(SOP_MESSAGE, ostr.str().c_str());
                }
                break;
            }
        }
    }

    // Combine two grids of the same type.
    template<typename GridT>
    void combineSameType()
    {
        typedef typename GridT::ValueType ValueT;

        const bool
            needA = self->needAGrid(op),
            needB = self->needBGrid(op);
        const float
            aMult = float(self->evalFloat("mult_a", 0, self->getTime())),
            bMult = float(self->evalFloat("mult_b", 0, self->getTime()));

        const GridT *aGrid = NULL, *bGrid = NULL;
        if (aBaseGrid) aGrid = UTvdbGridCast<GridT>(aBaseGrid).get();
        if (bBaseGrid) bGrid = UTvdbGridCast<GridT>(bBaseGrid).get();
        if (needA && !aGrid) throw std::runtime_error("missing A grid");
        if (needB && !bGrid) throw std::runtime_error("missing B grid");

        // Warn if combining vector grids with different vector types.
        if (needA && needB && openvdb::VecTraits<ValueT>::IsVec) {
            this->checkVectorTypes(aGrid, bGrid);
        }

        // If necessary, resample one grid so that its index space
        // registers with the other grid's.
        if (aGrid && bGrid) this->resampleGrids(aGrid, bGrid);

        const ValueT ZERO = openvdb::zeroVal<ValueT>();

        // A temporary grid is needed for binary operations, because they
        // cannibalize the B grid.
        typename GridT::Ptr resultGrid, tempGrid;

        switch (op) {
            case OP_COPY_A:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                break;

            case OP_COPY_B:
                MulAdd<GridT>(bMult).process(*bGrid, resultGrid);
                break;

            case OP_INVERT:
                MulAdd<GridT>(-aMult, 1.0).process(*aGrid, resultGrid);
                break;

            case OP_ADD:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                MulAdd<GridT>(bMult).process(*bGrid, tempGrid);
                openvdb::tools::compSum(*resultGrid, *tempGrid);
                break;

            case OP_SUBTRACT:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                MulAdd<GridT>(-bMult).process(*bGrid, tempGrid);
                openvdb::tools::compSum(*resultGrid, *tempGrid);
                break;

            case OP_MULTIPLY:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                MulAdd<GridT>(bMult).process(*bGrid, tempGrid);
                openvdb::tools::compMul(*resultGrid, *tempGrid);
                break;

            case OP_DIVIDE:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                MulAdd<GridT>(bMult).process(*bGrid, tempGrid);
                openvdb::tools::compDiv(*resultGrid, *tempGrid);
                break;

            case OP_MAXIMUM:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                MulAdd<GridT>(bMult).process(*bGrid, tempGrid);
                openvdb::tools::compMax(*resultGrid, *tempGrid);
                break;

            case OP_MINIMUM:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                MulAdd<GridT>(bMult).process(*bGrid, tempGrid);
                openvdb::tools::compMin(*resultGrid, *tempGrid);
                break;

            case OP_BLEND1: // (1 - A) * B
            {
                const Blend1<ValueT> comp(aMult, bMult);
                ValueT bg;
                comp(aGrid->background(), ZERO, bg);
#ifdef OPENVDB_3_ABI_COMPATIBLE
                resultGrid = aGrid->copy(/*tree=*/openvdb::CP_NEW);
#else
                resultGrid = aGrid->copyWithNewTree();
#endif
                openvdb::tools::changeBackground(resultGrid->tree(), bg);
                resultGrid->tree().combine2(aGrid->tree(), bGrid->tree(), comp, /*prune=*/false);
                break;
            }
            case OP_BLEND2: // A + (1 - A) * B
            {
                const Blend2<ValueT> comp(aMult, bMult);
                ValueT bg;
                comp(aGrid->background(), ZERO, bg);
#ifdef OPENVDB_3_ABI_COMPATIBLE
                resultGrid = aGrid->copy(/*tree=*/openvdb::CP_NEW);
#else
                resultGrid = aGrid->copyWithNewTree();
#endif
                openvdb::tools::changeBackground(resultGrid->tree(), bg);
                resultGrid->tree().combine2(aGrid->tree(), bGrid->tree(), comp, /*prune=*/false);
                break;
            }

            case OP_UNION:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                MulAdd<GridT>(bMult).process(*bGrid, tempGrid);
                openvdb::tools::csgUnion(*resultGrid, *tempGrid);
                break;

            case OP_INTERSECTION:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                MulAdd<GridT>(bMult).process(*bGrid, tempGrid);
                openvdb::tools::csgIntersection(*resultGrid, *tempGrid);
                break;

            case OP_DIFFERENCE:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                MulAdd<GridT>(bMult).process(*bGrid, tempGrid);
                openvdb::tools::csgDifference(*resultGrid, *tempGrid);
                break;

            case OP_REPLACE:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                MulAdd<GridT>(bMult).process(*bGrid, tempGrid);
                openvdb::tools::compReplace(*resultGrid, *tempGrid);
                break;

            case OP_TOPO_UNION:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                // Note: no need to scale the B grid for topology-only operations.
                resultGrid->topologyUnion(*bGrid);
                break;

            case OP_TOPO_INTERSECTION:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                resultGrid->topologyIntersection(*bGrid);
                break;

            case OP_TOPO_DIFFERENCE:
                MulAdd<GridT>(aMult).process(*aGrid, resultGrid);
                resultGrid->topologyDifference(*bGrid);
                break;
        }

        outGrid = this->postprocess<GridT>(resultGrid);
    }

    // Combine two grids of different types.
    /// @todo Currently, only topology operations can be performed on grids of different types.
    template<typename AGridT, typename BGridT>
    void combineDifferentTypes()
    {
        const bool
            needA = self->needAGrid(op),
            needB = self->needBGrid(op);

        const AGridT* aGrid = NULL;
        const BGridT* bGrid = NULL;
        if (aBaseGrid) aGrid = UTvdbGridCast<AGridT>(aBaseGrid).get();
        if (bBaseGrid) bGrid = UTvdbGridCast<BGridT>(bBaseGrid).get();
        if (needA && !aGrid) throw std::runtime_error("missing A grid");
        if (needB && !bGrid) throw std::runtime_error("missing B grid");

        // Warn if combining vector grids with different vector types.
        if (needA && needB && openvdb::VecTraits<typename AGridT::ValueType>::IsVec
            && openvdb::VecTraits<typename BGridT::ValueType>::IsVec)
        {
            this->checkVectorTypes(aGrid, bGrid);
        }

        // If necessary, resample one grid so that its index space
        // registers with the other grid's.
        if (aGrid && bGrid) this->resampleGrids(aGrid, bGrid);

        const float aMult = float(self->evalFloat("mult_a", 0, self->getTime()));

        typename AGridT::Ptr resultGrid;

        switch (op) {
            case OP_TOPO_UNION:
                MulAdd<AGridT>(aMult).process(*aGrid, resultGrid);
                // Note: no need to scale the B grid for topology-only operations.
                resultGrid->topologyUnion(*bGrid);
                break;

            case OP_TOPO_INTERSECTION:
                MulAdd<AGridT>(aMult).process(*aGrid, resultGrid);
                resultGrid->topologyIntersection(*bGrid);
                break;

            case OP_TOPO_DIFFERENCE:
                MulAdd<AGridT>(aMult).process(*aGrid, resultGrid);
                resultGrid->topologyDifference(*bGrid);
                break;

            default:
            {
                std::ostringstream ostr;
                ostr << "can't combine grid " << aGridName << " of type " << aGrid->type()
                    << "\n                 with grid " << bGridName
                    << " of type " << bGrid->type();
                throw std::runtime_error(ostr.str());
                break;
            }
        }

        outGrid = this->postprocess<AGridT>(resultGrid);
    }

    template<typename GridT>
    typename GridT::Ptr postprocess(typename GridT::Ptr resultGrid)
    {
        typedef typename GridT::ValueType ValueT;
        const ValueT ZERO = openvdb::zeroVal<ValueT>();

        const bool
            prune = self->evalInt("prune", 0, self->getTime()),
            flood = self->evalInt("flood", 0, self->getTime()),
            deactivate = self->evalInt("deactivate", 0, self->getTime());

        if (deactivate) {
            const float deactivationTolerance =
                float(self->evalFloat("bgtolerance", 0, self->getTime()));

            // Mark active output tiles and voxels as inactive if their
            // values match the output grid's background value.
            // Do this first to facilitate pruning.
            openvdb::tools::deactivate(*resultGrid, resultGrid->background(),
                ValueT(ZERO + deactivationTolerance));
        }

        if (flood && resultGrid->getGridClass() == openvdb::GRID_LEVEL_SET) {
            openvdb::tools::signedFloodFill(resultGrid->tree());
        }
        if (prune) {
            const float tolerance = float(self->evalFloat("tolerance", 0, self->getTime()));
            openvdb::tools::prune(resultGrid->tree(), ValueT(ZERO + tolerance));
        }

        return resultGrid;
    }

    template<typename AGridT>
    void operator()(typename AGridT::ConstPtr)
    {
        const bool
            needA = self->needAGrid(op),
            needB = self->needBGrid(op),
            needBoth = needA && needB;

        if (!needBoth || !aBaseGrid || !bBaseGrid || aBaseGrid->type() == bBaseGrid->type()) {
            this->combineSameType<AGridT>();
        } else {
            DispatchOp<AGridT> dispatcher(*this);
            // Dispatch on the B grid's type.
            int success = UTvdbProcessTypedGridTopology(
                UTvdbGetGridType(*bBaseGrid), bBaseGrid, dispatcher);
            if (!success) {
                std::ostringstream ostr;
                ostr << "grid " << bGridName << " has unsupported type " << bBaseGrid->type();
                self->addWarning(SOP_MESSAGE, ostr.str().c_str());
            }
        }
    }
}; // struct CombineOp


template<typename AGridT>
template<typename BGridT>
void
SOP_OpenVDB_Combine::DispatchOp<AGridT>::operator()(typename BGridT::ConstPtr)
{
    combineOp->combineDifferentTypes<AGridT, BGridT>();
}


////////////////////////////////////////


hvdb::GridPtr
SOP_OpenVDB_Combine::combineGrids(Operation op,
    hvdb::GridCPtr aGrid, hvdb::GridCPtr bGrid,
    const UT_String& aGridName, const UT_String& bGridName,
    ResampleMode resample)
{
    hvdb::GridPtr outGrid;

    const bool
        needA = needAGrid(op),
        needB = needBGrid(op),
        needLS = needLevelSets(op);

    if (!needA && !needB) throw std::runtime_error("nothing to do");
    if (needA && !aGrid) throw std::runtime_error("missing A grid");
    if (needB && !bGrid) throw std::runtime_error("missing B grid");

    if (needLS &&
        ((aGrid && aGrid->getGridClass() != openvdb::GRID_LEVEL_SET) ||
         (bGrid && bGrid->getGridClass() != openvdb::GRID_LEVEL_SET)))
    {
        std::ostringstream ostr;
        ostr << "expected level set grids for the " << SOP_OpenVDB_Combine::sOpMenuItems[op*2+1]
            << " operation,\n                 found "
            << hvdb::Grid::gridClassToString(aGrid->getGridClass()) << " (" << aGridName << ") and "
            << hvdb::Grid::gridClassToString(bGrid->getGridClass()) << " (" << bGridName
            << ");\n                 the output grid will not be a valid level set";
        addWarning(SOP_MESSAGE, ostr.str().c_str());
    }

    if (needA && needB && aGrid->type() != bGrid->type()
        && op != OP_TOPO_UNION && op != OP_TOPO_INTERSECTION && op != OP_TOPO_DIFFERENCE)
    {
        std::ostringstream ostr;
        ostr << "can't combine grid " << aGridName << " of type " << aGrid->type()
            << "\n                 with grid " << bGridName << " of type " << bGrid->type();
        addWarning(SOP_MESSAGE, ostr.str().c_str());
        return outGrid;
    }

    CombineOp compOp;
    compOp.self = this;
    compOp.op = op;
    compOp.resample = resample;
    compOp.aBaseGrid = aGrid;
    compOp.bBaseGrid = bGrid;
    compOp.aGridName = aGridName;
    compOp.bGridName = bGridName;
    compOp.interrupt = hvdb::Interrupter();

    int success = UTvdbProcessTypedGridTopology(
        UTvdbGetGridType(needA ? *aGrid : *bGrid), aGrid, compOp);
    if (!success || !compOp.outGrid) {
        std::ostringstream ostr;
        if (aGrid->type() == bGrid->type()) {
            ostr << "grids " << aGridName << " and " << bGridName
                << " have unsupported type " << aGrid->type();
        } else {
            ostr << "grid " << (needA ? aGridName : bGridName)
                << " has unsupported type " << (needA ? aGrid->type() : bGrid->type());
        }
        addWarning(SOP_MESSAGE, ostr.str().c_str());
    }
    return compOp.outGrid;
}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
