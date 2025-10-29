#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -ex
IMATH_VERSION="$1"

git clone https://github.com/AcademySoftwareFoundation/Imath.git
cd Imath

if [ "$IMATH_VERSION" != "latest" ]; then
    git checkout tags/${IMATH_VERSION} -b ${IMATH_VERSION}
fi

mkdir build
cd build

cmake ..

make -j$(nproc)
sudo make install
