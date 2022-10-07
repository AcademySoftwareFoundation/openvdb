#!/usr/bin/env bash

set -ex

GTEST_VERSION="$1"

git clone https://github.com/google/googletest.git
cd googletest

if [ "$GTEST_VERSION" != "latest" ]; then
    git checkout release-${GTEST_VERSION} -b v${GTEST_VERSION}
fi

mkdir build
cd build
cmake ../.
make -j8
make install
