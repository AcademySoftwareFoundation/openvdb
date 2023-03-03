#!/usr/bin/env bash

set -ex

vcpkg update
vcpkg install zlib libpng openexr tbb gtest blosc glfw3 glew python3 \
    boost-iostreams boost-any boost-uuid boost-interprocess boost-algorithm pybind11 \
    --clean-after-build
