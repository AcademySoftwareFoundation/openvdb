#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -ex

GTEST_VERSION="$1"


git clone https://github.com/google/googletest.git -b v${GTEST_VERSION}
cd googletest
mkdir build
cd build
cmake ..

make -j$(nproc)

sudo make install
