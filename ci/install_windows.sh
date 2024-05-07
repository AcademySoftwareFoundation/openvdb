#!/usr/bin/env bash

set -ex

vcpkg update
vcpkg install zlib libpng openexr tbb gtest cppunit blosc glfw3 glew python3 jemalloc \
    boost-iostreams boost-interprocess boost-algorithm pybind11 \
    --clean-after-build
