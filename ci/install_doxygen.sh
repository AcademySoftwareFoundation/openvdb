#!/usr/bin/env bash

set -ex

DOXYGEN_VERSION="$1"

git clone https://github.com/doxygen/doxygen.git
cd doxygen

if [ "$DOXYGEN_VERSION" != "latest" ]; then
    git checkout Release_${DOXYGEN_VERSION} -b v${DOXYGEN_VERSION}
fi

mkdir build
cd build
cmake ../.
make -j8
make install
