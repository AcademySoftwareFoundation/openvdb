// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/math/QuantizedUnitVec.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Vec3.h>

#include <gtest/gtest.h>

#include <sstream>
#include <algorithm>
#include <cmath>
#include <ctime>


class TestQuantizedUnitVec: public ::testing::Test
{
protected:
    // Generate a random number in the range [0, 1].
    double randNumber() { return double(rand()) / (double(RAND_MAX) + 1.0); }
};


////////////////////////////////////////


namespace {
const uint16_t
    MASK_XSIGN = 0x8000, // 1000000000000000
    MASK_YSIGN = 0x4000, // 0100000000000000
    MASK_ZSIGN = 0x2000; // 0010000000000000
}


////////////////////////////////////////


TEST_F(TestQuantizedUnitVec, testQuantization)
{
    using namespace openvdb;
    using namespace openvdb::math;

    //
    // Check sign bits
    //
    Vec3s unitVec = Vec3s(-1.0, -1.0, -1.0);
    unitVec.normalize();

    uint16_t quantizedVec = QuantizedUnitVec::pack(unitVec);

    EXPECT_TRUE((quantizedVec & MASK_XSIGN));
    EXPECT_TRUE((quantizedVec & MASK_YSIGN));
    EXPECT_TRUE((quantizedVec & MASK_ZSIGN));

    unitVec[0] = -unitVec[0];
    unitVec[2] = -unitVec[2];
    quantizedVec = QuantizedUnitVec::pack(unitVec);

    EXPECT_TRUE(!(quantizedVec & MASK_XSIGN));
    EXPECT_TRUE((quantizedVec & MASK_YSIGN));
    EXPECT_TRUE(!(quantizedVec & MASK_ZSIGN));

    unitVec[1] = -unitVec[1];
    quantizedVec = QuantizedUnitVec::pack(unitVec);

    EXPECT_TRUE(!(quantizedVec & MASK_XSIGN));
    EXPECT_TRUE(!(quantizedVec & MASK_YSIGN));
    EXPECT_TRUE(!(quantizedVec & MASK_ZSIGN));

    QuantizedUnitVec::flipSignBits(quantizedVec);

    EXPECT_TRUE((quantizedVec & MASK_XSIGN));
    EXPECT_TRUE((quantizedVec & MASK_YSIGN));
    EXPECT_TRUE((quantizedVec & MASK_ZSIGN));

    unitVec[2] = -unitVec[2];
    quantizedVec = QuantizedUnitVec::pack(unitVec);
    QuantizedUnitVec::flipSignBits(quantizedVec);

    EXPECT_TRUE((quantizedVec & MASK_XSIGN));
    EXPECT_TRUE((quantizedVec & MASK_YSIGN));
    EXPECT_TRUE(!(quantizedVec & MASK_ZSIGN));

    //
    // Check conversion error
    //
    const double tol = 0.05; // component error tolerance

    const int numNormals = 40000;


    // init
    srand(0);
    const int n = int(std::sqrt(double(numNormals)));
    const double xScale = (2.0 * openvdb::math::pi<double>()) / double(n);
    const double yScale = openvdb::math::pi<double>() / double(n);

    double x, y, theta, phi;
    Vec3s n0, n1;

    // generate random normals, by uniformly distributing points on a unit-sphere.

    // loop over a [0 to n) x [0 to n) grid.
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {

            // jitter, move to random pos. inside the current cell
            x = double(a) + randNumber();
            y = double(b) + randNumber();

            // remap to a lat/long map
            theta = y * yScale; // [0 to PI]
            phi   = x * xScale; // [0 to 2PI]

            // convert to cartesian coordinates on a unit sphere.
            // spherical coordinate triplet (r=1, theta, phi)
            n0[0] = float(std::sin(theta)*std::cos(phi));
            n0[1] = float(std::sin(theta)*std::sin(phi));
            n0[2] = float(std::cos(theta));

            EXPECT_NEAR(1.0, n0.length(), 1e-6);

            n1 = QuantizedUnitVec::unpack(QuantizedUnitVec::pack(n0));

            EXPECT_NEAR(1.0, n1.length(), 1e-6);

            EXPECT_NEAR(n0[0], n1[0], tol);
            EXPECT_NEAR(n0[1], n1[1], tol);
            EXPECT_NEAR(n0[2], n1[2], tol);

            float sumDiff = std::abs(n0[0] - n1[0]) + std::abs(n0[1] - n1[1])
                + std::abs(n0[2] - n1[2]);

            EXPECT_TRUE(sumDiff < (2.0 * tol));
        }
    }
}
