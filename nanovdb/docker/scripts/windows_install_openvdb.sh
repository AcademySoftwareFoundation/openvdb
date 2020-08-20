#!/usr/bin/env bash

set -ex

vcpkg update
vcpkg install zlib tbb blosc
# This fails due to openvdb requiring openexr (which is broken in vcpkg)
#vcpkg install openvdb[tools] --recurse
