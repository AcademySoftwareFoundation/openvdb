#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -ex
EXR_VERSION="$1"

git clone https://github.com/AcademySoftwareFoundation/openexr.git
cd openexr

if [ "$EXR_VERSION" != "latest" ]; then
    git checkout tags/${EXR_VERSION} -b ${EXR_VERSION}
fi

mkdir build
cd build

cmake ..

make -j$(nproc)
sudo make install
