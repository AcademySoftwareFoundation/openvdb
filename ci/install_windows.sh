#!/usr/bin/env bash

set -ex

vcpkg update
vcpkg install zlib openexr tbb gtest blosc glfw3 glew
