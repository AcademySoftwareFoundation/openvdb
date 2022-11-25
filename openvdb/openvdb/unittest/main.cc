// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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

#include <gtest/gtest.h>

int
main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
