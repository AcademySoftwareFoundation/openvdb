// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// A simple minimal unit test to serve as a starting point for fVDB unit tests.

#include <torch/torch.h>

#include <gtest/gtest.h>

TEST(Example, ExampleTest) {
    std::size_t const size = 100;
    EXPECT_TRUE(torch::equal(torch::diagonal(torch::eye(size)), torch::ones(size)));
}
