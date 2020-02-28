#!/usr/bin/env bash

set -ex

vcpkg install zlib openexr tbb cppunit blosc
vcpkg integrate install
