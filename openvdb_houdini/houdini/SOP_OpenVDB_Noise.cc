///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Noise.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Applies noise to level sets represented by VDBs. The noise can
/// optionally be masked by another level set

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/math/Operators.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/tools/Interpolation.h> // for box sampler
#include <UT/UT_PNoise.h>
#include <UT/UT_Interrupt.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;
namespace cvdb = openvdb;


namespace { // anon namespace

struct FractalBoltzmanGenerator
{
    FractalBoltzmanGenerator(float freq, float amp, int octaves, float gain,
        float lacunarity, float roughness, int mode):
        mOctaves(octaves), mNoiseMode(mode), mFreq(freq), mAmp(amp), mGain(gain),
        mLacunarity(lacunarity), mRoughness(roughness)
    {}

    // produce the noise as float
    float noise(cvdb::Vec3R point, float freqMult = 1.0f) const
    {
        float signal;
        float result = 0.0f;
        float curamp = mAmp;
        float curfreq = mFreq * freqMult;

        for (int n = 0; n <= mOctaves; n++) {
            point = (point * curfreq);

            // convert to float for UT_PNoise
            float location[3] = { float(point[0]), float(point[1]), float(point[2]) };

            // generate noise in the [-1,1] range
            signal = 2.0f*UT_PNoise::noise3D(location) - 1.0f;

            if (mNoiseMode > 0) {
                signal = cvdb::math::Pow(cvdb::math::Abs(signal), mGain);
            }

            result  += (signal * curamp);
            curfreq = mLacunarity;
            curamp *= mRoughness;
        }
        if (mNoiseMode == 1) {
            result = -result;
        }

        return result;
    }

private:
    // member data
    int mOctaves;
    int mNoiseMode;
    float mFreq;
    float mAmp;
    float mGain;
    float mLacunarity;
    float mRoughness;

};

struct NoiseSettings
{
    NoiseSettings() : mMaskMode(0), mOffset(0.0), mThreshold(0.0),
        mFallOff(0.0), mNOffset(cvdb::Vec3R(0.0, 0.0, 0.0)) {}

    int mMaskMode;
    float mOffset, mThreshold, mFallOff;
    cvdb::Vec3R mNOffset;
};

} // anon namespace


////////////////////////////////////////


class SOP_OpenVDB_Noise: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Noise(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Noise() {}

    static OP_Node* factory(OP_Network*, const char*, OP_Operator*);

    virtual int isRefInput(unsigned input) const { return (input == 1); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

private:
    // Process the given grid and return the output grid.
    // Can be applied to FloatGrid or DoubleGrid
    // this contains the majority of the work
    template<typename GridType>
    void applyNoise(hvdb::Grid& grid, const FractalBoltzmanGenerator&,
        const NoiseSettings&, const hvdb::Grid* maskGrid) const;

    bool mSecondInputConnected;
};


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Define a string-valued group name pattern parameter and add it to the list.
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to be processed.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    // amplitude
    parms.add(hutil::ParmFactory(PRM_FLT_J, "amp", "Amplitude")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_FREE, 10.0));

    // frequency
    parms.add(hutil::ParmFactory(PRM_FLT_J, "freq", "Frequency")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_FREE, 1.0));

    // Octaves
    parms.add(hutil::ParmFactory(PRM_INT_J, "oct", "Octaves")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_FREE, 10));

    // Lacunarity
    parms.add(hutil::ParmFactory(PRM_FLT_J, "lac", "Lacunarity")
        .setDefault(PRMtwoDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_FREE, 10.0));

    // Gain
    parms.add(hutil::ParmFactory(PRM_FLT_J, "gain", "Gain")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_FREE, 1.0));

    // Roughness
    parms.add(hutil::ParmFactory(PRM_FLT_J, "rough", "Roughness")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 10.0));

    // SurfaceOffset
    parms.add(hutil::ParmFactory(PRM_FLT_J, "soff", "Surface Offset")
        .setDefault(PRMzeroDefaults));

    // Noise Offset
    parms.add(hutil::ParmFactory(PRM_XYZ_J,"noff", "Noise Offset")
        .setVectorSize(3)
        .setDefault(PRMzeroDefaults));

    {   // Noise Mode
        const char* items[] = {
            "straight", "Straight",
            "abs",      "Absolute",
            "invabs",   "Inverse Absolute",
            NULL
        };
        parms.add(hutil::ParmFactory(PRM_ORD, "mode", "Noise Mode")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    // Mask {
    parms.add(hutil::ParmFactory(PRM_HEADING, "maskHeading", "Maks"));

    // Group
    parms.add(
        hutil::ParmFactory(PRM_STRING, "maskGroup",  "Mask Group")
        .setChoiceList(&hutil::PrimGroupMenuInput2));

    {   // Use mask
        const char* items[] = {
            "maskless",       "No noise if mask < threshold",
            "maskgreater",    "No noise if mask > threshold",
            "maskgreaternml", "No noise if mask > threshold & normals align",
            "maskisfreqmult", "Use mask as frequency multiplier",
            NULL
        };
        parms.add(hutil::ParmFactory(PRM_ORD, "mask", "Mask")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    // mask threshold
    parms.add(hutil::ParmFactory(PRM_FLT_J, "thres", "Mask Threshold")
        .setDefault(PRMzeroDefaults));

    // Fall off
    parms.add(hutil::ParmFactory(PRM_FLT_J, "fall", "Fall-Off")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_FREE, 10.0));
    // }

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Noise", SOP_OpenVDB_Noise::factory, parms, *table)
        .addAlias("OpenVDB LevelSet Noise")
        .addInput("VDB grids to noise")
        .addOptionalInput("Optional VDB grid to use as mask");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Noise::factory(OP_Network* net, const char* name, OP_Operator *op)
{
    return new SOP_OpenVDB_Noise(net, name, op);
}


SOP_OpenVDB_Noise::SOP_OpenVDB_Noise(OP_Network* net,
                                                     const char* name,
                                                     OP_Operator* op):
    SOP_NodeVDB(net, name, op),
    mSecondInputConnected(false)
{
    UT_PNoise::initNoise();
}


////////////////////////////////////////


bool
SOP_OpenVDB_Noise::updateParmsFlags()
{

    bool changed = false;

    changed |= enableParm("maskGroup", mSecondInputConnected);
    changed |= enableParm("mask", mSecondInputConnected);
    changed |= enableParm("thres", mSecondInputConnected);
    changed |= enableParm("fall", mSecondInputConnected);

    changed |= setVisibleState("maskHeading", mSecondInputConnected);
    changed |= setVisibleState("maskGroup", mSecondInputConnected);
    changed |= setVisibleState("mask", mSecondInputConnected);
    changed |= setVisibleState("thres", mSecondInputConnected);
    changed |= setVisibleState("fall", mSecondInputConnected);

    return changed;
}


////////////////////////////////////////


template<typename GridType>
void
SOP_OpenVDB_Noise::applyNoise(
    hvdb::Grid& grid,
    const FractalBoltzmanGenerator& fbGenerator,
    const NoiseSettings& settings,
    const hvdb::Grid* mask) const
{
    // Use second order finite difference.
    typedef cvdb::math::Gradient<cvdb::math::GenericMap, cvdb::math::CD_2ND>  Gradient;
    typedef cvdb::math::CPT<cvdb::math::GenericMap, cvdb::math::CD_2ND>       CPT;
    typedef cvdb::math::SecondOrderDenseStencil<GridType>                     StencilType;

    typedef typename GridType::TreeType                     TreeType;
    typedef cvdb::math::Vec3<typename TreeType::ValueType>  Vec3Type;

    // Down cast the generic pointer to the output grid.
    GridType& outGrid = UTvdbGridCast<GridType>(grid);

    const cvdb::math::Transform& xform = grid.transform();

    // Create a stencil.
    StencilType stencil(outGrid); // uses its own grid accessor

    // scratch variables
    typename GridType::ValueType result; // result - use mask as frequency multiplier
    cvdb::Vec3R voxelPt; // voxel coordinates
    cvdb::Vec3R worldPt; // world coordinates
    float noise, alpha;

    // The use of the GenericMap is a performance compromise
    // because the GenericMap holdds a base class pointer.
    // This should be optimized by resolving the acutal map type
    cvdb::math::GenericMap map(grid);

    if (!mask) {
        alpha = 1.0f;
        for (typename GridType::ValueOnIter v = outGrid.beginValueOn(); v; ++v) {
            stencil.moveTo(v);
            worldPt = xform.indexToWorld(CPT::result(map, stencil) + settings.mNOffset);
            noise = fbGenerator.noise(worldPt);
            v.setValue(*v + alpha * (noise - settings.mOffset));
        }
        return;
    }

    // Down cast the generic pointer to the mask grid.
    const GridType* maskGrid = UTvdbGridCast<GridType>(mask);
    const cvdb::math::Transform& maskXform = mask->transform();

    switch (settings.mMaskMode) {
        case 0: //No noise if mask < threshold
        {
            for (typename GridType::ValueOnIter v = outGrid.beginValueOn(); v; ++v) {
                cvdb::Coord ijk = v.getCoord();
                stencil.moveTo(ijk); // in voxel units

                worldPt = xform.indexToWorld(ijk);
                voxelPt = maskXform.worldToIndex(worldPt);
                cvdb::tools::BoxSampler::sample<TreeType>(maskGrid->tree(), voxelPt, result);

                // apply threshold
                if (result < settings.mThreshold) {
                    continue; //next voxel
                }
                alpha = static_cast<float>(result >= settings.mThreshold + settings.mFallOff ?
                    1.0f : (result - settings.mThreshold) / settings.mFallOff);

                worldPt = xform.indexToWorld(CPT::result(map, stencil) + settings.mNOffset);
                noise = fbGenerator.noise(worldPt);
                v.setValue(*v + alpha * (noise - settings.mOffset));
            }
        }
        break;

        case 1: //No noise if mask > threshold
        {
            for (typename GridType::ValueOnIter v = outGrid.beginValueOn(); v; ++v) {
                cvdb::Coord ijk = v.getCoord();
                stencil.moveTo(ijk); // in voxel units

                worldPt = xform.indexToWorld(ijk);
                voxelPt = maskXform.worldToIndex(worldPt);
                cvdb::tools::BoxSampler::sample<TreeType>(maskGrid->tree(), voxelPt, result);

                // apply threshold
                if (result > settings.mThreshold) {
                    continue; //next voxel
                }
                alpha = static_cast<float>(result <= settings.mThreshold - settings.mFallOff ?
                    1.0f : (settings.mThreshold - result) / settings.mFallOff);

                worldPt = xform.indexToWorld(CPT::result(map, stencil) + settings.mNOffset);
                noise = fbGenerator.noise(worldPt);
                v.setValue(*v + alpha * (noise - settings.mOffset));
            }
        }
        break;

        case 2: //No noise if mask < threshold & normals align
        {
            StencilType maskStencil(*maskGrid);
            for (typename GridType::ValueOnIter v = outGrid.beginValueOn(); v; ++v) {
                cvdb::Coord ijk = v.getCoord();
                stencil.moveTo(ijk); // in voxel units

                worldPt = xform.indexToWorld(ijk);
                voxelPt = maskXform.worldToIndex(worldPt);
                cvdb::tools::BoxSampler::sample<TreeType>(maskGrid->tree(), voxelPt, result);

                // for the gradient of the maskGrid
                cvdb::Coord mask_ijk((int)voxelPt[0], (int)voxelPt[1], (int)voxelPt[2]);
                maskStencil.moveTo(mask_ijk);
                // normal alignment
                Vec3Type grid_grad = Gradient::result(map, stencil);
                Vec3Type mask_grad = Gradient::result(map, maskStencil);
                const double c = cvdb::math::Abs(grid_grad.dot(mask_grad));

                if (result > settings.mThreshold && c > 0.9) continue;//next voxel
                alpha = static_cast<float>(result <= settings.mThreshold - settings.mFallOff ?
                    1.0f : (settings.mThreshold - result) / settings.mFallOff);

                worldPt = xform.indexToWorld(CPT::result(map, stencil) + settings.mNOffset);
                noise = fbGenerator.noise(worldPt);
                v.setValue(*v + alpha * (noise - settings.mOffset));
            }
        }
        break;

        case 3: //Use mask as frequency multiplier
        {
            alpha = 1.0f;
            for (typename GridType::ValueOnIter v = outGrid.beginValueOn(); v; ++v) {
                cvdb::Coord ijk = v.getCoord();
                stencil.moveTo(ijk); // in voxel units

                worldPt = xform.indexToWorld(ijk);
                voxelPt = maskXform.worldToIndex(worldPt);
                cvdb::tools::BoxSampler::sample<TreeType>(maskGrid->tree(), voxelPt, result);

                worldPt = xform.indexToWorld(CPT::result(map, stencil) + settings.mNOffset);
                // Use result of sample as frequency multiplier.
                noise = fbGenerator.noise(worldPt, static_cast<float>(result));
                v.setValue(*v + alpha * (noise - settings.mOffset));
            }
        }
        break;

        default: // should never get here
            throw std::runtime_error("internal error in mode selection");
    }// end switch
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Noise::cookMySop(OP_Context &context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal time = context.getTime();

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSourceStealable(0, context);

        // Evaluate the FractalBoltzman noise parameters from UI
        FractalBoltzmanGenerator fbGenerator(static_cast<float>(evalFloat("freq", 0, time)),
                                             static_cast<float>(evalFloat("amp", 0, time)),
                                             evalInt("oct", 0, time),
                                             static_cast<float>(evalFloat("gain", 0, time)),
                                             static_cast<float>(evalFloat("lac", 0, time)),
                                             static_cast<float>(evalFloat("rough", 0, time)),
                                             evalInt("mode", 0, time));

        NoiseSettings settings;

        // evaluate parameter for blending noise
        settings.mOffset = static_cast<float>(evalFloat("soff", 0, time));
        settings.mNOffset = cvdb::Vec3R(evalFloat("noff", 0, time),
                                        evalFloat("noff", 1, time),
                                        evalFloat("noff", 2, time));

        // Mask
        const openvdb::GridBase* maskGrid = NULL;
        const GU_Detail* refGdp = inputGeo(1);
        mSecondInputConnected = refGdp != NULL;
        UT_String groupStr;

        if (mSecondInputConnected) {
            evalString(groupStr, "maskGroup", 0, time);

            const GA_PrimitiveGroup* maskGroup =
                matchGroup(const_cast<GU_Detail&>(*refGdp), groupStr.toStdString());

            hvdb::VdbPrimCIterator gridIter(refGdp, maskGroup);

            if (gridIter) {
                settings.mMaskMode = evalInt("mask", 0, time);
                settings.mThreshold = static_cast<float>(evalFloat("thres", 0, time));
                settings.mFallOff = static_cast<float>(evalFloat("fall", 0, time));

                maskGrid = &((*gridIter)->getGrid());
                ++gridIter;
            }

            if (gridIter) {
                std::ostringstream ostr;
                ostr << "Found more than one grid in the mask group; the first grid will be used.";
                addWarning(SOP_MESSAGE, ostr.str().c_str());
            }
        }

        // Do the work..
        UT_AutoInterrupt progress("OpenVDB LS Noise");

        // Get the group of grids to process.
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        // For each VDB primitive in the selected group.
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {

            if (progress.wasInterrupted()) {
                throw std::runtime_error("processing was interrupted");
            }

            GU_PrimVDB* vdbPrim = *it;

            if (vdbPrim->getStorageType() == UT_VDB_FLOAT) {
                vdbPrim->makeGridUnique();
                applyNoise<cvdb::ScalarGrid>(vdbPrim->getGrid(), fbGenerator, settings, maskGrid);

            } else if (vdbPrim->getStorageType() == UT_VDB_DOUBLE) {
                vdbPrim->makeGridUnique();
                applyNoise<cvdb::DoubleGrid>(vdbPrim->getGrid(), fbGenerator, settings, maskGrid);

            } else {
                std::stringstream ss;
                ss << "VDB primitive " << it.getPrimitiveNameOrIndex()
                    << " was skipped because it is not a scalar grid.";
                addWarning(SOP_MESSAGE, ss.str().c_str());
                continue;
            }
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
