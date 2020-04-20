#!/usr/bin/env bash

set -ex

vcpkg update
vcpkg install zlib openexr tbb cppunit blosc glfw3 glew
