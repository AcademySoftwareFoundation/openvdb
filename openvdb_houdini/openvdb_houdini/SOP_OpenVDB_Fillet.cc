// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
// @file SOP_OpenVDB_Fillet.cc
//
// @author FX R&D OpenVDB team
//
// @brief OpenVDB Fillet to combine two level-sets together.

#include <iostream>
#include <stdexcept>
#include <string>

#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h> // for isExactlyEqual()
#include <openvdb/openvdb.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/GridTransformer.h> // for resampleToMatch()
#include <openvdb/tools/LevelSetRebuild.h> // for levelSetRebuild()
#include <openvdb/tools/Blend.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/UT_VDBUtils.h>
#include <openvdb_houdini/Utils.h>

#include <houdini_utils/ParmFactory.h>
#include <UT/UT_Interrupt.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

namespace {

//
// Resampling options
//
enum ResampleMode {
    RESAMPLE_OFF,    // don't auto-resample grids
    RESAMPLE_B,      // resample B to match A
    RESAMPLE_A,      // resample A to match B
    RESAMPLE_HI_RES, // resample higher-res grid to match lower-res
    RESAMPLE_LO_RES  // resample lower-res grid to match higher-res
};


enum { RESAMPLE_MODE_FIRST = RESAMPLE_OFF, RESAMPLE_MODE_LAST = RESAMPLE_LO_RES };


//
// VDBFilletParms struct to be used in SOP
//
struct VDBFilletParms {
    float mAlpha = 10.f; // Falloff
    float mBeta = 100.f; // Exponent
    float mGamma = 10.f; // Amplitude/Multiplier
    ResampleMode mResampleMode = ResampleMode::RESAMPLE_OFF; // Resample mode
    int mSamplingOrder = 0;
    hvdb::GridCPtr mABaseGrid = nullptr;
    hvdb::GridCPtr mBBaseGrid = nullptr;
    openvdb::FloatGrid::ConstPtr mMaskPtr = nullptr;
};

// Helper function to compare both scalar and vector values
template <typename ValueT>
static bool approxEq(const ValueT& a, const ValueT& b) {
    return openvdb::math::isRelOrApproxEqual(a, b, ValueT(1e-6f), ValueT(1e-8f));
}

const char* const sResampleModeMenuItems[] = {
    "off",      "Off",
    "btoa",     "B to Match A",
    "atob",     "A to Match B",
    "hitolo",   "Higher-res to Match Lower-res",
    "lotohi",   "Lower-res to Match Higher-res",
    nullptr
};

inline ResampleMode
asResampleMode(exint i, ResampleMode defaultMode = RESAMPLE_B)
{
    return (i >= RESAMPLE_MODE_FIRST && i <= RESAMPLE_MODE_LAST)
        ? static_cast<ResampleMode>(i) : defaultMode;
}

using StringVec = std::vector<std::string>;

// Split a string into group patterns separated by whitespace.
// For example, given '@name=d* @id="1 2" {grp1 grp2}', return
// ['@name=d*', '@id="1 2"', '{grp1 grp2}'].
// (This is nonstandard.  Normally, multiple patterns are unioned
// to define a single group.)
// Nesting of quotes and braces is not supported.
inline StringVec
splitPatterns(const std::string& str)
{
    StringVec patterns;
    bool quoted = false, braced = false;
    std::string pattern;
    for (const auto c: str) {
        if (isspace(c)) {
            if (pattern.empty()) continue; // skip whitespace between patterns
            if (quoted || braced) {
                pattern.push_back(c); // keep whitespace within quotes or braces
            } else {
                // At the end of a pattern.  Start a new pattern.
                patterns.push_back(pattern);
                pattern.clear();
                quoted = braced = false;
            }
        } else {
            switch (c) {
                case '"': quoted = !quoted; break;
                case '{': braced = true; break;
                case '}': braced = false; break;
                default: break;
            }
            pattern.push_back(c);
        }
    }
    if (!pattern.empty()) { patterns.push_back(pattern); } // add the final pattern

    // If no patterns were found, add an empty pattern, which matches everything.
    if (patterns.empty()) { patterns.push_back(""); }

    return patterns;
}

inline UT_String
getGridName(const GU_PrimVDB* vdb, const UT_String& defaultName = "")
{
    UT_String name{UT_String::ALWAYS_DEEP};
    if (vdb != nullptr) {
        name = vdb->getGridName();
        if (!name.isstring()) name = defaultName;
    }
    return name;
}

template<typename GridT>
struct ResampleOp {
    ResampleOp(typename GridT::ConstPtr &aGrid, typename GridT::ConstPtr &bGrid, const VDBFilletParms &parms)
              : mAGrid(aGrid)
              , mBGrid(bGrid)
              , mParms(parms)
              { }

    // Function to return a scalar grid's background value as a double
    static double getScalarBackgroundValue(const hvdb::Grid& baseGrid)
    {
        double bg = 0.0;
        baseGrid.apply<hvdb::NumericGridTypes>([&bg](const auto &grid) {
            bg = static_cast<double>(grid.background());
        });
        return bg;
    }

    typename GridT::Ptr resampleToMatch(const GridT& src, const hvdb::Grid& ref, int order)
    {
        using ValueT = typename GridT::ValueType;
        const ValueT ZERO = openvdb::zeroVal<ValueT>();

        const openvdb::math::Transform& refXform = ref.constTransform();

        typename GridT::Ptr dest;
        if (src.getGridClass() == openvdb::GRID_LEVEL_SET) {
            // For level set grids, use the level set rebuild tool to both resample the
            // source grid to match the reference grid and to rebuild the resulting level set.
            const bool refIsLevelSet = ref.getGridClass() == openvdb::GRID_LEVEL_SET;
            OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
            const ValueT halfWidth = refIsLevelSet
                ? ValueT(ZERO + getScalarBackgroundValue(ref) * (1.0 / ref.voxelSize()[0]))
                : ValueT(src.background() * (1.0 / src.voxelSize()[0]));
            OPENVDB_NO_TYPE_CONVERSION_WARNING_END

            if (!openvdb::math::isFinite(halfWidth)) {
                std::stringstream msg;
                msg << "Resample to match: Illegal narrow band width = " << halfWidth
                    << ", caused by grid '" << ref.getName() << "' with background "
                    << getScalarBackgroundValue(ref);
                throw std::invalid_argument(msg.str());
            }

            try {
                dest = openvdb::tools::levelSetRebuild(src, /*iso=*/ZERO,
                    /*exWidth=*/halfWidth, /*inWidth=*/halfWidth, &refXform/*, &mInterrupt.interrupter*/);
            } catch (openvdb::TypeError&) {
                dest.reset();
                std::stringstream msg;
                msg << "skipped rebuild of level set grid " << src.getName() << " of type " << src.type();
                throw std::runtime_error(msg.str().c_str());
            }
        }
        if (!dest && src.constTransform() != refXform) {
            // If level set rebuild failed due to an unsupported grid type,
            // use the grid transformer tool to resample the source grid to match
            // the reference grid.
            dest = src.copyWithNewTree();
            dest->setTransform(refXform.copy());
            using namespace openvdb;
            switch (order) {
            case 0: tools::resampleToMatch<tools::PointSampler>(src, *dest/*, mInterrupt.interrupter()*/); break;
            case 1: tools::resampleToMatch<tools::BoxSampler>(src, *dest/*, mInterrupt.interrupter()*/); break;
            case 2: tools::resampleToMatch<tools::QuadraticSampler>(src, *dest/*, mInterrupt.interrupter()*/); break;
            // note: no default case because sampling order is guaranteed to be 0, 1, or 2 in evalParms.
            }
        }
        return dest;
    }

    void resampleMask(typename GridT::ConstPtr& mask)
    {
        if (!mask) return;
        const openvdb::math::Transform& maskXform = mask->constTransform();
        const openvdb::math::Transform& aGrdXform = mAGrid->constTransform();
        if (maskXform != aGrdXform) {
            typename GridT::ConstPtr maskRsmpl = this->resampleToMatch(*mask /* src */, *mAGrid /* ref */, mParms.mSamplingOrder);
            mask = maskRsmpl;
        }
    }

    void resampleGrids()
    {
        if (!mAGrid || !mBGrid) return;

        // One of RESAMPLE_A, RESAMPLE_B or RESAMPLE_OFF, specifying whether
        // grid A, grid B or neither grid was resampled
        int resampleWhich = RESAMPLE_OFF;

        // Determine which of the two grids should be resampled.
        if (mParms.mResampleMode == RESAMPLE_HI_RES || mParms.mResampleMode == RESAMPLE_LO_RES) {
            const openvdb::Vec3d
                aVoxSize = mAGrid->voxelSize(),
                bVoxSize = mBGrid->voxelSize();
            const double
                aVoxVol = aVoxSize[0] * aVoxSize[1] * aVoxSize[2],
                bVoxVol = bVoxSize[0] * bVoxSize[1] * bVoxSize[2];
            resampleWhich = ((aVoxVol > bVoxVol && mParms.mResampleMode == RESAMPLE_LO_RES)
                || (aVoxVol < bVoxVol && mParms.mResampleMode == RESAMPLE_HI_RES))
                ? RESAMPLE_A : RESAMPLE_B;
        } else {
            resampleWhich = mParms.mResampleMode;
        }

        if (mAGrid->constTransform() != mBGrid->constTransform()) {
            // If the A and B grid transforms don't match, one of the grids
            // should be resampled into the other's index space.
            if (mParms.mResampleMode == RESAMPLE_OFF) {
                // Resampling is disabled.  Just log a warning.
                std::ostringstream msg;
                msg << mAGrid->getName() << " and " << mBGrid->getName() << " transforms don't match";
                throw std::runtime_error(msg.str().c_str());
            } else {
                if (resampleWhich == RESAMPLE_A) {
                    // Resample grid A into grid B's index space.
                    mAGrid = this->resampleToMatch(*mAGrid, *mBGrid, mParms.mSamplingOrder);
                } else if (resampleWhich == RESAMPLE_B) {
                    // Resample grid B into grid A's index space.
                    mBGrid = this->resampleToMatch(*mBGrid, *mAGrid, mParms.mSamplingOrder);
                }
            }
        }

        // If both grids are level sets, ensure that their background values match.
        // (If one of the grids was resampled, then the background values should
        // already match.)
        if (mAGrid->getGridClass() == openvdb::GRID_LEVEL_SET &&
            mBGrid->getGridClass() == openvdb::GRID_LEVEL_SET)
        {
            const double
                a = this->getScalarBackgroundValue(*mAGrid),
                b = this->getScalarBackgroundValue(*mBGrid);
            if (!approxEq<double>(a, b)) {
                if (mParms.mResampleMode == RESAMPLE_OFF) {
                    // Resampling/rebuilding is disabled.  Just log a warning.
                    std::ostringstream msg;
                    msg << mAGrid->getName() << " and " << mBGrid->getName()
                        << " background values don't match ("
                        << std::setprecision(3) << a << " vs. " << b << ");\n"
                        << "                 the output grid will not be a valid level set";
                    throw std::runtime_error(msg.str().c_str());
                } else {
                    // One of the two grids needs a level set rebuild.
                    if (resampleWhich == RESAMPLE_A) {
                        // Rebuild A to match B's background value.
                        mAGrid = this->resampleToMatch(*mAGrid, *mBGrid, mParms.mSamplingOrder);
                    } else if (resampleWhich == RESAMPLE_B) {
                        // Rebuild B to match A's background value.
                        mBGrid = this->resampleToMatch(*mBGrid, *mAGrid, mParms.mSamplingOrder);
                    }
                }
            }
        }
    }

    typename GridT::ConstPtr &mAGrid;
    typename GridT::ConstPtr &mBGrid;
    const VDBFilletParms mParms;
    ResampleMode mResampleMode;
    int mSamplingOrder;
};
}

class SOP_OpenVDB_Fillet: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Fillet(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Fillet() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    class Cache: public SOP_VDBCacheOptions
    {
    protected:
        OP_ERROR cookVDBSop(OP_Context&) override;
        OP_ERROR evalParms(OP_Context&, VDBFilletParms&);
    private:
        hvdb::GridPtr blendLevelSets(
            hvdb::GridCPtr aGrid,
            hvdb::GridCPtr bGrid,
            VDBFilletParms parm);
    }; // class Cache

    // Return true for a given input if the connector to the input
    // should be drawn dashed rather than solid.
    int isRefInput(unsigned idx) const override { return (idx == 1); }

protected:
    unsigned disableParms() override;
};


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Fillet::Cache::evalParms(OP_Context& context, VDBFilletParms& parms)
{
    const fpreal time = context.getTime();

    parms.mAlpha = static_cast<float>(evalFloat("alpha", 0, time));
    parms.mBeta  = static_cast<float>(evalFloat("beta", 0, time));
    parms.mGamma = static_cast<float>(evalFloat("gamma", 0, time));
    parms.mResampleMode = asResampleMode(evalInt("resample", 0, time));
    parms.mSamplingOrder = static_cast<int>(evalInt("resampleinterp", 0, time));

    if (parms.mSamplingOrder < 0 || parms.mSamplingOrder > 2) {
        addWarning(SOP_MESSAGE, "Sampling order should be 0, 1, or 2.");
        return UT_ERROR_ABORT;
    }

    // mask
    int useMask = static_cast<int>(evalInt("mask", 0, time));

    openvdb::FloatGrid::ConstPtr maskGrid;

    if (useMask && this->nInputs() == 3) {
        const GU_Detail* maskGeo = inputGeo(2);
        const auto maskName = evalStdString("maskname", time);
        if (!maskGeo) {
            addWarning(SOP_MESSAGE, "The mask input is empty.");
            return UT_ERROR_ABORT;
        }

        const GA_PrimitiveGroup* maskGroup = parsePrimitiveGroups(maskName.c_str(), GroupCreator(maskGeo));
        if (!maskGroup && !maskName.empty()) {
            addWarning(SOP_MESSAGE, "Mask not found.");
            return UT_ERROR_ABORT;
        }

        hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);
        if (!maskIt) {
            addWarning(SOP_MESSAGE, "The mask input is empty.");
            return UT_ERROR_ABORT;
        } else if (maskIt->getStorageType() != UT_VDB_FLOAT) {
            addWarning(SOP_MESSAGE, "The mask grid has to be a Float Grid.");
            return UT_ERROR_ABORT;
        }

        maskGrid = openvdb::gridConstPtrCast<openvdb::FloatGrid>(maskIt->getGridPtr());
        if (!maskGrid) {
            addWarning(SOP_MESSAGE, "The mask grid has to be a Float Grid.");
            return UT_ERROR_ABORT;
        }

        parms.mMaskPtr = maskGrid;

    } else if (useMask && this->nInputs() < 3) {
        addWarning(SOP_MESSAGE, "Need a mask grid as a third input to SOP.");
        return UT_ERROR_ABORT;
    } // if not using mask, do nothing.

    return error();
}


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    // Group A
    parms.add(hutil::ParmFactory(PRM_STRING, "agroup", "Group A")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Use a subset of the first input as the A VDB(s).")
        .setDocumentation(
            "The VDBs to be used from the first input"
            " (see [specifying volumes|/model/volumes#group])"));

    // Group B
    parms.add(hutil::ParmFactory(PRM_STRING, "bgroup", "Group B")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setTooltip("Use a subset of the second input as the B VDB(s).")
        .setDocumentation(
            "The VDBs to be used from the second input"
            " (see [specifying volumes|/model/volumes#group])"));

    // Mask multiplier
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "mask", "")
        .setDefault(PRMzeroDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setTooltip("Enable / disable the mask."));

    parms.add(hutil::ParmFactory(PRM_STRING, "maskname", "Alpha Mask")
        .setChoiceList(&hutil::PrimGroupMenuInput3)
        .setTooltip("Optional scalar VDB used for alpha masking\n\n"
            "Values are assumed to be between 0 and 1."));

    // Band radius of influence / falloff width, i.e. alpha
    parms.add(hutil::ParmFactory(PRM_FLT_J, "alpha", "Band Radius")
        .setDefault(10.f)
        .setRange(PRM_RANGE_UI, 0.f, PRM_RANGE_UI, 1000.f)
        .setTooltip(
            "Band radius of influence measures the distance from the zero\n"
            "iso-contour of the intersection that is going to be modified.\n"
            "This is measured in world-space."));

    // Exponent, i.e. beta
    parms.add(hutil::ParmFactory(PRM_FLT_J, "beta", "Exponent")
        .setDefault(100.f)
        .setRange(PRM_RANGE_UI, 0.f, PRM_RANGE_UI, 1000.f)
        .setTooltip(
            "Blending curve exponential used in the model."));

    // Amplitude
    parms.add(hutil::ParmFactory(PRM_FLT_J, "gamma", "Multiplier")
        .setDefault(10.f)
        .setRange(PRM_RANGE_UI, 0.f, PRM_RANGE_UI, 1000.f)
        .setTooltip(
            "Amplitude provides a multiplier to make the blended\n"
            "influence weaker or stronger."));

    // Menu of resampling options
    parms.add(hutil::ParmFactory(PRM_ORD, "resample", "Resample")
        .setDefault(PRMoneDefaults)
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, sResampleModeMenuItems)
        .setTooltip(
            "If the A and B VDBs have different transforms, one VDB should\n"
            "be resampled to match the other before the two are combined.\n"
            "Also, level set VDBs should have matching background values\n"
            "(i.e., matching narrow band widths)."));

    // Menu of resampling interpolation order options
    parms.add(hutil::ParmFactory(PRM_ORD, "resampleinterp", "Interpolation")
        .setDefault(PRMoneDefaults)
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "point",     "Nearest",
            "linear",    "Linear",
            "quadratic", "Quadratic"
        })
        .setTooltip(
            "Specify the type of interpolation to be used when\n"
            "resampling one VDB to match the other's transform.")
        .setDocumentation(
            "The type of interpolation to be used when resampling one VDB"
            " to match the other's transform\n\n"
            "Nearest neighbor interpolation is fast but can introduce noticeable"
            " sampling artifacts.  Quadratic interpolation is slow but high-quality."
            " Linear interpolation is intermediate in speed and quality."));


    // Register this operator.
    // (See houdini_utils/Utils.h for OpFactory details.)
    hvdb::OpenVDBOpFactory("VDB Fillet", SOP_OpenVDB_Fillet::factory, parms, *table)
        .addInput("A VDBs")
        .addOptionalInput("B VDBs")
        .addOptionalInput("Mask VDB")
        .setVerb(SOP_NodeVerb::COOK_INPLACE, []() { return new SOP_OpenVDB_Fillet::Cache; });
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Fillet::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Fillet(net, name, op);
}


SOP_OpenVDB_Fillet::SOP_OpenVDB_Fillet(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


// Enable/disable or show/hide parameters in the UI.
unsigned
SOP_OpenVDB_Fillet::disableParms()
{
    unsigned changed = 0;

    // Disable parms.
    //changed += enableParm("dummy",
    //    evalInt("dummy", 0, /*time=*/0));


    //setVisibleState("dummy", getEnableState("dummy"));

    return changed;
}


////////////////////////////////////////

hvdb::GridPtr
SOP_OpenVDB_Fillet::Cache::blendLevelSets(
    hvdb::GridCPtr aGrid,
    hvdb::GridCPtr bGrid,
    VDBFilletParms parms)
{
    openvdb::FloatGrid::ConstPtr a = openvdb::gridConstPtrCast<openvdb::FloatGrid>(aGrid);
    openvdb::FloatGrid::ConstPtr b = openvdb::gridConstPtrCast<openvdb::FloatGrid>(bGrid);

    // sanitizers
    if (!a) throw std::runtime_error("Missing the first grid");
    if (!b) throw std::runtime_error("Missing the second grid");
    if (a->getGridClass() != openvdb::GRID_LEVEL_SET)
        throw std::runtime_error("First grid needs to be a level-set.");
    if (b->getGridClass() != openvdb::GRID_LEVEL_SET)
        throw std::runtime_error("Second grid needs to be a level-set.");

    ResampleOp<openvdb::FloatGrid> rsmpl(a, b, parms);
    rsmpl.resampleGrids();
    if (parms.mMaskPtr) rsmpl.resampleMask(parms.mMaskPtr);

    hvdb::GridPtr ret = openvdb::tools::unionFillet<openvdb::FloatGrid, typename openvdb::FloatGrid>(
        *a, *b, parms.mMaskPtr, parms.mAlpha, parms.mBeta, parms.mGamma);

    return ret;
}

OP_ERROR
SOP_OpenVDB_Fillet::Cache::cookVDBSop(OP_Context& context)
{
    try {
        UT_AutoInterrupt progress{"Processing VDB grids"};

        const fpreal time = context.getTime();

        // Get gdps
        GU_Detail* aGdp = gdp;
        const GU_Detail* bGdp = inputGeo(1, context);

        // Get the group of grids to process.
        const auto aGroupStr = evalStdString("agroup", time);
        const auto bGroupStr = evalStdString("bgroup", time);
        const auto* bGroup = (!bGdp ?  nullptr : matchGroup(*bGdp, bGroupStr));

        // Fill-in VDBFilletParms
        VDBFilletParms parms;
        if (evalParms(context, parms) >= UT_ERROR_ABORT) return error();

        // The collation pattern that we follow is 'pairs' as in
        // SOP_OpenVDB_Combine. This means to combine pairs of A and B VDBs
        // in the order in which they appear in their respective groups.
        std::vector<const GA_PrimitiveGroup*> aGroupVec;
        for (const auto& pattern: splitPatterns(aGroupStr)) {
            aGroupVec.push_back(matchGroup(*aGdp, pattern));
        }

        // Iterate over one or more A groups.
        for (const auto* aGroup: aGroupVec) {
            hvdb::VdbPrimIterator aIt{aGdp, GA_Range::safedeletions{}, aGroup};
            hvdb::VdbPrimCIterator bIt{bGdp, bGroup};

            // Populate two vectors of primitives, one comprising the A grids
            // and the other the B grids. (In the case of flattening operations,
            // these grids might be taken from the same input.)
            // Note: the following relies on exhausted iterators returning nullptr
            // and on incrementing an exhausted iterator being a no-op.
            std::vector<GU_PrimVDB*> aVdbVec;
            std::vector<const GU_PrimVDB*> bVdbVec;
            for ( ; aIt && bIt; ++aIt, ++bIt) {
                aVdbVec.push_back(*aIt);
                bVdbVec.push_back(*bIt);
            }

            std::set<GU_PrimVDB*> vdbsToRemove;
            // Iterate over A and, optionally, B grids.
            for (size_t i = 0, N = std::min(aVdbVec.size(), bVdbVec.size()); i < N; ++i) {
                if (progress.wasInterrupted()) { throw std::runtime_error{"interrupted"}; }

                // Note: even if needA is false, we still need to delete A grids.
                GU_PrimVDB* aVdb = aVdbVec[i];
                const GU_PrimVDB* bVdb = bVdbVec[i];

                hvdb::GridPtr aGrid;
                hvdb::GridCPtr bGrid;
                if (aVdb) aGrid = aVdb->getGridPtr();
                if (bVdb) bGrid = bVdb->getConstGridPtr();

                parms.mABaseGrid = aGrid;
                parms.mBBaseGrid = bGrid;

                if (hvdb::GridPtr outGrid = blendLevelSets(aGrid, bGrid, parms))
                {
                    // Name the output grid after the A grid if the A grid is used,
                    // or after the B grid otherwise.
                    UT_String outGridName = getGridName(aVdb);
                    // Add a new VDB primitive for the output grid to the output gdp.
                    GU_PrimVDB::buildFromGrid(*gdp, outGrid, /*copyAttrsFrom=*/ aVdb, outGridName);
                    vdbsToRemove.insert(aVdb);
                }
            } // iterate over aVdbVec and bVdbVec

            // Remove primitives that were copied from input 0.
            for (GU_PrimVDB* vdb: vdbsToRemove) {
                if (vdb) gdp->destroyPrimitive(*vdb, /*andPoints=*/true);
            }
        } // aGroup : aGroupVec
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}
