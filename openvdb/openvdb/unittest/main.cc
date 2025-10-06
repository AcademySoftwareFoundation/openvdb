// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm> // for std::shuffle()
#include <cmath> // for std::round()
#include <cstdlib> // for EXIT_SUCCESS
#include <cstring> // for strrchr()
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#if defined(__linux__) && defined(OPENVDB_TESTS_FPE)
#include <fenv.h>
#endif

#include <gtest/gtest.h>

int
main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

#if defined(__linux__) && defined(OPENVDB_TESTS_FPE)
    int excepts = FE_DIVBYZERO | FE_INVALID;
    if (getenv("OPENVDB_TEST_OVERFLOW")) {
        // when set, test for FP overflow as well.
        excepts = excepts | FE_OVERFLOW;
    }
    feenableexcept(excepts);
#endif

    return RUN_ALL_TESTS();
}
