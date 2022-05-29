#!/usr/bin/env bash

set -ex

vcpkg update
vcpkg install zlib libpng openexr tbb gtest blosc glfw3 glew \
    boost-iostreams boost-any boost-uuid boost-interprocess boost-algorithm \
    --clean-after-build
