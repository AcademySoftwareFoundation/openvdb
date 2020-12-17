// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file GridManagerUtils.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of GridManagerUtils.
*/

#define _USE_MATH_DEFINES
#include <cmath>

#include "primitives.h"

namespace viewer {
namespace primitives {

nanovdb::Vec4f mandelbulb(const nanovdb::Vec3f& p)
{
    using namespace nanovdb;
    auto  w = p;
    float m = w.lengthSqr();

    auto  color = Vec4f(Abs(w[0]), Abs(w[1]), Abs(w[2]), m);
    float dz = 1.0f;

    for (int i = 0; i < 4; i++) {
        //dz = 8.0f * powf(sqrtf(m), 7.0f) * dz + 1.0f;
        dz = 8.0 * pow(m, 3.5) * dz + 1.0;

        float r = w.length();
        float b = 8.0f * acosf(w[1] / r);
        float a = 8.0f * atan2f(w[0], w[2]);
        w = p +
            powf(r, 8.0f) * Vec3f(sinf(b) * sinf(a), cosf(b), sinf(b) * cosf(a));

        color[0] = Min(color[0], Abs(w[0]));
        color[1] = Min(color[1], Abs(w[1]));
        color[2] = Min(color[2], Abs(w[2]));
        color[3] = Min(color[3], m);

        m = w.lengthSqr();
        if (m > 256.0)
            break;
    }

    return Vec4f(color[0], color[1], color[2], 0.25f * logf(m) * sqrtf(m) / dz);
}

}
} // namespace viewer::primitives