#!/usr/bin/env bash

set -ex

vcpkg update
vcpkg install zlib openexr tbb gtest blosc glfw3 glew \
    boost-iostreams boost-system boost-any boost-uuid boost-interprocess boost-algorithm
