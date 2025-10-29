#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -ex
GTEST_VERSION="$1"

git clone https://github.com/google/googletest.git
cd googletest

if [ "$GTEST_VERSION" != "latest" ]; then
    git checkout tags/${GTEST_VERSION} -b ${GTEST_VERSION}
fi

mkdir build
cd build

cmake ..

make -j$(nproc)
sudo make install
