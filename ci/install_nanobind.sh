#!/usr/bin/env bash

set -ex

PYBIND11_VERSION="$1"
INSTALL_ROOT="$2"
CMAKE_EXTRA=()
if [ ! -z "${INSTALL_ROOT}" ]; then
    CMAKE_EXTRA+=("-DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}")
fi

git clone --recurse-submodules https://github.com/wjakob/nanobind.git
cd nanobind

if [ "$PYBIND11_VERSION" != "latest" ]; then
    git checkout tags/v${PYBIND11_VERSION} -b v${PYBIND11_VERSION}
fi

mkdir build
cd build

cmake \
    -DNB_TEST=OFF \
    "${CMAKE_EXTRA[@]}" \
    ..

make -j8
sudo make install
