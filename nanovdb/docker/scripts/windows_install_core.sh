#!/usr/bin/env bash

set -ex

vcpkg update
vcpkg install zlib tbb blosc
