// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Jeff Lait
///
/// @date  May 13, 2021
///
/// @file DitherLUT.h
///
/// @brief Defines look up table to do dithering of 8^3 leaf nodes.

#ifndef NANOVDB_DITHERLUT_HAS_BEEN_INCLUDED
#define NANOVDB_DITHERLUT_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>// for __hostdev__, Vec3, Min, Max, Pow2, Pow3, Pow4

namespace nanovdb {

namespace math {

class DitherLUT
{
    const bool mEnable;
public:
    /// @brief Constructor with an optional scaling factor for the dithering
    __hostdev__ DitherLUT(bool enable = true) : mEnable(enable) {}

    /// @brief Retrieves dither threshold for an offset within an 8^3 leaf nodes.
    ///
    /// @param offset into the lookup table of size 512
    __hostdev__ float operator()(const int offset)
    {

// This table was generated with
/**************

static constexpr inline uint32
SYSwang_inthash(uint32 key)
{
    // From http://www.concentric.net/~Ttwang/tech/inthash.htm
    key += ~(key << 16);
    key ^=  (key >> 5);
    key +=  (key << 3);
    key ^=  (key >> 13);
    key += ~(key << 9);
    key ^=  (key >> 17);
    return key;
}

static void
ut_initDitherR(float *pattern, float offset,
    int x, int y, int z, int res, int goalres)
{
    // These offsets are designed to maximize the difference between
    // dither values in nearby voxels within a given 2x2x2 cell, without
    // producing axis-aligned artifacts.  The are organized in row-major
    // order.
    static const float  theDitherOffset[] = {0,4,6,2,5,1,3,7};
    static const float  theScale = 0.125F;
    int         key = (((z << res) + y) << res) + x;

    if (res == goalres)
    {
    pattern[key] = offset;
    return;
    }

    // Randomly flip (on each axis) the dithering patterns used by the
    // subcells.  This key is xor'd with the subcell index below before
    // looking up in the dither offset list.
    key = SYSwang_inthash(key) & 7;

    x <<= 1;
    y <<= 1;
    z <<= 1;

    offset *= theScale;
    for (int i = 0; i < 8; i++)
    ut_initDitherR(pattern, offset+theDitherOffset[i ^ key]*theScale,
        x+(i&1), y+((i&2)>>1), z+((i&4)>>2), res+1, goalres);
}

// This is a compact algorithm that accomplishes essentially the same thing
// as ut_initDither() above.  We should eventually switch to use this and
// clean the dead code.
static fpreal32 *
ut_initDitherRecursive(int goalres)
{
    const int nfloat = 1 << (goalres*3);
    float   *pattern = new float[nfloat];
    ut_initDitherR(pattern, 1.0F, 0, 0, 0, 0, goalres);

    // This has built an even spacing from 1/nfloat to 1.0.
    // however, our dither pattern should be 1/(nfloat+1) to nfloat/(nfloat+1)
    // So we do a correction here.  Note that the earlier calculations are
    // done with powers of 2 so are exact, so it does make sense to delay
    // the renormalization to this pass.
    float correctionterm = nfloat / (nfloat+1.0F);
    for (int i = 0; i < nfloat; i++)
        pattern[i] *= correctionterm;
    return pattern;
}

    theDitherMatrix = ut_initDitherRecursive(3);

    for (int i = 0; i < 512/8; i ++)
    {
        for (int j = 0; j < 8; j ++)
            std::cout << theDitherMatrix[i*8+j] << "f, ";
        std::cout << std::endl;
    }

 **************/
        static const float LUT[512] =
        {
            0.14425f, 0.643275f, 0.830409f, 0.331384f, 0.105263f, 0.604289f, 0.167641f, 0.666667f,
            0.892788f, 0.393762f, 0.0818713f, 0.580897f, 0.853801f, 0.354776f, 0.916179f, 0.417154f,
            0.612086f, 0.11306f, 0.79922f, 0.300195f, 0.510721f, 0.0116959f, 0.947368f, 0.448343f,
            0.362573f, 0.861598f, 0.0506823f, 0.549708f, 0.261209f, 0.760234f, 0.19883f, 0.697856f,
            0.140351f, 0.639376f, 0.576998f, 0.0779727f, 0.522417f, 0.0233918f, 0.460039f, 0.959064f,
            0.888889f, 0.389864f, 0.327485f, 0.826511f, 0.272904f, 0.77193f, 0.709552f, 0.210526f,
            0.483431f, 0.982456f, 0.296296f, 0.795322f, 0.116959f, 0.615984f, 0.0545809f, 0.553606f,
            0.732943f, 0.233918f, 0.545809f, 0.0467836f, 0.865497f, 0.366472f, 0.803119f, 0.304094f,
            0.518519f, 0.0194932f, 0.45614f, 0.955166f, 0.729045f, 0.230019f, 0.54191f, 0.042885f,
            0.269006f, 0.768031f, 0.705653f, 0.206628f, 0.479532f, 0.978558f, 0.292398f, 0.791423f,
            0.237817f, 0.736842f, 0.424951f, 0.923977f, 0.136452f, 0.635478f, 0.323587f, 0.822612f,
            0.986355f, 0.487329f, 0.674464f, 0.175439f, 0.88499f, 0.385965f, 0.573099f, 0.0740741f,
            0.51462f, 0.0155945f, 0.202729f, 0.701754f, 0.148148f, 0.647174f, 0.834308f, 0.335283f,
            0.265107f, 0.764133f, 0.951267f, 0.452242f, 0.896686f, 0.397661f, 0.08577f, 0.584795f,
            0.8577f, 0.358674f, 0.920078f, 0.421053f, 0.740741f, 0.241715f, 0.678363f, 0.179337f,
            0.109162f, 0.608187f, 0.17154f, 0.670565f, 0.491228f, 0.990253f, 0.42885f, 0.927875f,
            0.0662768f, 0.565302f, 0.62768f, 0.128655f, 0.183236f, 0.682261f, 0.744639f, 0.245614f,
            0.814815f, 0.315789f, 0.378168f, 0.877193f, 0.931774f, 0.432749f, 0.495127f, 0.994152f,
            0.0350877f, 0.534113f, 0.97076f, 0.471735f, 0.214425f, 0.71345f, 0.526316f, 0.0272904f,
            0.783626f, 0.2846f, 0.222222f, 0.721248f, 0.962963f, 0.463938f, 0.276803f, 0.775828f,
            0.966862f, 0.467836f, 0.405458f, 0.904483f, 0.0701754f, 0.569201f, 0.881092f, 0.382066f,
            0.218324f, 0.717349f, 0.654971f, 0.155945f, 0.818713f, 0.319688f, 0.132554f, 0.631579f,
            0.0623782f, 0.561404f, 0.748538f, 0.249513f, 0.912281f, 0.413255f, 0.974659f, 0.475634f,
            0.810916f, 0.311891f, 0.499025f, 0.998051f, 0.163743f, 0.662768f, 0.226121f, 0.725146f,
            0.690058f, 0.191033f, 0.00389864f, 0.502924f, 0.557505f, 0.0584795f, 0.120858f, 0.619883f,
            0.440546f, 0.939571f, 0.752437f, 0.253411f, 0.307992f, 0.807018f, 0.869396f, 0.37037f,
            0.658869f, 0.159844f, 0.346979f, 0.846004f, 0.588694f, 0.0896686f, 0.152047f, 0.651072f,
            0.409357f, 0.908382f, 0.596491f, 0.0974659f, 0.339181f, 0.838207f, 0.900585f, 0.401559f,
            0.34308f, 0.842105f, 0.779727f, 0.280702f, 0.693957f, 0.194932f, 0.25731f, 0.756335f,
            0.592593f, 0.0935673f, 0.0311891f, 0.530214f, 0.444444f, 0.94347f, 0.506823f, 0.00779727f,
            0.68616f, 0.187135f, 0.124756f, 0.623782f, 0.288499f, 0.787524f, 0.350877f, 0.849903f,
            0.436647f, 0.935673f, 0.873294f, 0.374269f, 0.538012f, 0.0389864f, 0.60039f, 0.101365f,
            0.57115f, 0.0721248f, 0.758285f, 0.259259f, 0.719298f, 0.220273f, 0.532164f, 0.0331384f,
            0.321637f, 0.820663f, 0.00974659f, 0.508772f, 0.469786f, 0.968811f, 0.282651f, 0.781676f,
            0.539961f, 0.0409357f, 0.727096f, 0.22807f, 0.500975f, 0.00194932f, 0.563353f, 0.0643275f,
            0.290448f, 0.789474f, 0.477583f, 0.976608f, 0.251462f, 0.750487f, 0.31384f, 0.812865f,
            0.94152f, 0.442495f, 0.879142f, 0.380117f, 0.37232f, 0.871345f, 0.309942f, 0.808967f,
            0.192982f, 0.692008f, 0.130604f, 0.62963f, 0.621832f, 0.122807f, 0.559454f, 0.0604289f,
            0.660819f, 0.161793f, 0.723197f, 0.224172f, 0.403509f, 0.902534f, 0.840156f, 0.341131f,
            0.411306f, 0.910331f, 0.473684f, 0.97271f, 0.653021f, 0.153996f, 0.0916179f, 0.590643f,
            0.196881f, 0.695906f, 0.384016f, 0.883041f, 0.0955166f, 0.594542f, 0.157895f, 0.65692f,
            0.945419f, 0.446394f, 0.633528f, 0.134503f, 0.844055f, 0.345029f, 0.906433f, 0.407407f,
            0.165692f, 0.664717f, 0.103314f, 0.602339f, 0.126706f, 0.625731f, 0.189084f, 0.688109f,
            0.91423f, 0.415205f, 0.851852f, 0.352827f, 0.875244f, 0.376218f, 0.937622f, 0.438596f,
            0.317739f, 0.816764f, 0.255361f, 0.754386f, 0.996101f, 0.497076f, 0.933723f, 0.434698f,
            0.567251f, 0.0682261f, 0.504873f, 0.00584795f, 0.247563f, 0.746589f, 0.185185f, 0.684211f,
            0.037037f, 0.536062f, 0.0994152f, 0.598441f, 0.777778f, 0.278752f, 0.465887f, 0.964912f,
            0.785575f, 0.28655f, 0.847953f, 0.348928f, 0.0292398f, 0.528265f, 0.7154f, 0.216374f,
            0.39961f, 0.898636f, 0.961014f, 0.461988f, 0.0487329f, 0.547758f, 0.111111f, 0.610136f,
            0.649123f, 0.150097f, 0.212476f, 0.711501f, 0.797271f, 0.298246f, 0.859649f, 0.360624f,
            0.118908f, 0.617934f, 0.0565302f, 0.555556f, 0.329435f, 0.82846f, 0.516569f, 0.0175439f,
            0.867446f, 0.368421f, 0.805068f, 0.306043f, 0.578947f, 0.079922f, 0.267057f, 0.766082f,
            0.270955f, 0.76998f, 0.707602f, 0.208577f, 0.668616f, 0.169591f, 0.606238f, 0.107212f,
            0.520468f, 0.0214425f, 0.45809f, 0.957115f, 0.419103f, 0.918129f, 0.356725f, 0.855751f,
            0.988304f, 0.489279f, 0.426901f, 0.925926f, 0.450292f, 0.949318f, 0.512671f, 0.0136452f,
            0.239766f, 0.738791f, 0.676413f, 0.177388f, 0.699805f, 0.20078f, 0.263158f, 0.762183f,
            0.773879f, 0.274854f, 0.337232f, 0.836257f, 0.672515f, 0.173489f, 0.734893f, 0.235867f,
            0.0253411f, 0.524366f, 0.586745f, 0.0877193f, 0.423002f, 0.922027f, 0.48538f, 0.984405f,
            0.74269f, 0.243665f, 0.680312f, 0.181287f, 0.953216f, 0.454191f, 0.1423f, 0.641326f,
            0.493177f, 0.992203f, 0.430799f, 0.929825f, 0.204678f, 0.703704f, 0.890838f, 0.391813f,
            0.894737f, 0.395712f, 0.0838207f, 0.582846f, 0.0448343f, 0.54386f, 0.231969f, 0.730994f,
            0.146199f, 0.645224f, 0.832359f, 0.333333f, 0.793372f, 0.294347f, 0.980507f, 0.481481f,
            0.364522f, 0.863548f, 0.80117f, 0.302144f, 0.824561f, 0.325536f, 0.138402f, 0.637427f,
            0.614035f, 0.11501f, 0.0526316f, 0.551657f, 0.0760234f, 0.575049f, 0.88694f, 0.387914f,
        };
        return mEnable ? LUT[offset & 511] : 0.5f;// branch prediction should optimize this!
    }
}; // DitherLUT class

}// namspace math

}// namespace nanovdb

#endif // NANOVDB_DITHERLUT_HAS_BEEN_INCLUDED
