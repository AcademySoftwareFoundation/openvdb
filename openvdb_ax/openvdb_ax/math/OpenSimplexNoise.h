// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file math/OpenSimplexNoise.h
///
/// @authors Francisco Gochez
///
/// @brief  Methods for generating OpenSimplexNoise (n-dimensional gradient noise)
///
/// @details  This code is based on https://gist.github.com/tombsar/716134ec71d1b8c1b530
///   (accessed on 22/05/2019). We have simplified that code in a number of ways,
///   most notably by removing the template on dimension (this only generates 3
///   dimensional noise) and removing the base class as it's unnecessary for our
///   uses. We also assume C++ 2011 or above and have thus removed a number of
///   ifdef blocks.
///
///   The OSN namespace contains the original copyright.
///

#ifndef OPENVDB_AX_MATH_OPEN_SIMPLEX_NOISE_HAS_BEEN_INCLUDED
#define OPENVDB_AX_MATH_OPEN_SIMPLEX_NOISE_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <cstdint>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace math {

template <typename NoiseT>
void curlnoise(double (*out)[3], const double (*in)[3])
{
    float delta = 0.0001f;
    float a, b;

    // noise coordinates for vector potential components.
    float p[3][3] = {
        { static_cast<float>((*in)[0]) + 000.0f, static_cast<float>((*in)[1]) + 000.0f, static_cast<float>((*in)[2]) + 000.0f }, // x
        { static_cast<float>((*in)[0]) + 256.0f, static_cast<float>((*in)[1]) - 256.0f, static_cast<float>((*in)[2]) + 256.0f }, // y
        { static_cast<float>((*in)[0]) - 512.0f, static_cast<float>((*in)[1]) + 512.0f, static_cast<float>((*in)[2]) - 512.0f }, // z
    };

    OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
    // Compute curl.x
    a = (NoiseT::noise(p[2][0], p[2][1] + delta, p[2][2]) - NoiseT::noise(p[2][0], p[2][1] - delta, p[2][2])) / (2.0f * delta);
    b = (NoiseT::noise(p[1][0], p[1][1], p[1][2] + delta) - NoiseT::noise(p[1][0], p[1][1], p[1][2] - delta)) / (2.0f * delta);
    (*out)[0] = a - b;

    // Compute curl.y
    a = (NoiseT::noise(p[0][0], p[0][1], p[0][2] + delta) - NoiseT::noise(p[0][0], p[0][1], p[0][2] - delta)) / (2.0f * delta);
    b = (NoiseT::noise(p[2][0] + delta, p[2][1], p[2][2]) - NoiseT::noise(p[2][0] - delta, p[2][1], p[2][2])) / (2.0f * delta);
    (*out)[1] = a - b;

    // Compute curl.z
    a = (NoiseT::noise(p[1][0] + delta, p[1][1], p[1][2]) - NoiseT::noise(p[1][0] - delta, p[1][1], p[1][2])) / (2.0f * delta);
    b = (NoiseT::noise(p[0][0], p[0][1] + delta, p[0][2]) - NoiseT::noise(p[0][0], p[0][1] - delta, p[0][2])) / (2.0f * delta);
    (*out)[2] = a - b;
    OPENVDB_NO_TYPE_CONVERSION_WARNING_END
}

template <typename NoiseT>
void curlnoise(double (*out)[3], double x, double y, double z)
{
    const double in[3] = {x, y, z};
    curlnoise<NoiseT>(out, &in);
}

}
}
}
}

namespace OSN
{

// The following is the original copyright notice:
/*
 *
 *
 * OpenSimplex (Simplectic) Noise in C++
 * by Arthur Tombs
 *
 * Modified 2015-01-08
 *
 * This is a derivative work based on OpenSimplex by Kurt Spencer:
 *   https://gist.github.com/KdotJPG/b1270127455a94ac5d19
 *
 * Anyone is free to make use of this software in whatever way they want.
 * Attribution is appreciated, but not required.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

// 3D Implementation of the OpenSimplexNoise generator.
class OPENVDB_AX_API OSNoise
{
public:
    using inttype = int64_t;

    // Initializes the class using a permutation array generated from a 64-bit seed.
    // Generates a proper permutation (i.e. doesn't merely perform N successive
    // pair swaps on a base array).
    // Uses a simple 64-bit LCG.
    OSNoise(inttype seed = 0LL);

    OSNoise(const int * p);

    template <typename T>
    T eval(const T x, const T y, const T z) const;

private:

    template <typename T>
    inline T extrapolate(const inttype xsb,
                         const inttype ysb,
                         const inttype zsb,
                         const T dx,
                         const T dy,
                         const T dz) const;

    template <typename T>
    inline T extrapolate(const inttype xsb,
                         const inttype ysb,
                         const inttype zsb,
                         const T dx,
                         const T dy,
                         const T dz,
                         T (&de) [3]) const;

    int mPerm [256];
    // Array of gradient values for 3D. Values are defined below the class definition.
    static const int sGradients [72];

    // Because 72 is not a power of two, extrapolate cannot use a bitmask to index
    // into the perm array. Pre-calculate and store the indices instead.
    int mPermGradIndex [256];
};

}

#endif // OPENVDB_AX_MATH_OPEN_SIMPLEX_NOISE_HAS_BEEN_INCLUDED
