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
    int excepts = FE_DIVBYZERO;
    // Note: NO FE_OVERFLOW as some unit tests don't pass with that.
    // Note: NO FE_INVALID as some unit tests don't pass with that.
    feenableexcept(excepts);
#endif

    return RUN_ALL_TESTS();
}
