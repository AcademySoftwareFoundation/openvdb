#!/usr/bin/env bash

set -ex

BLOSC_VERSION="$1"
INSTALL_ROOT="$2"
CMAKE_EXTRA=()
if [ ! -z "${INSTALL_ROOT}" ]; then
    CMAKE_EXTRA+=("-DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}")
fi

git clone https://github.com/Blosc/c-blosc.git
cd c-blosc

if [ "$BLOSC_VERSION" != "latest" ]; then
    git checkout tags/v${BLOSC_VERSION} -b v${BLOSC_VERSION}
fi

mkdir build
cd build

# On MacOS there's a bug between blosc 1.14-1.20 where unistd isn't included
# in zlib-1.2.8/gzlib.c. Provide -DPREFER_EXTERNAL_ZLIB to use the installed
# version of zlib.
# https://github.com/Blosc/python-blosc/issues/229
cmake \
    -DPREFER_EXTERNAL_ZLIB=ON \
    -DBUILD_STATIC=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_FUZZERS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    "${CMAKE_EXTRA[@]}" \
    ..

make -j8
make install
