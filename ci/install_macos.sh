#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -x

# Remove Python3 symlinks in /usr/local/bin as workaround to brew update issues
# https://github.com/actions/setup-python/issues/577
rm /usr/local/bin/2to3* || :
rm /usr/local/bin/idle3* || :
rm /usr/local/bin/pydoc* || :
rm /usr/local/bin/python3* || :

brew update
brew install bash gnu-getopt # for CI scripts
brew install c-blosc
brew install cmake
brew install glfw
brew install googletest
brew install jq # for trivial parsing of brew json
brew install openexr
brew install nanobind # also installs the dependent python version
brew install robin-map # required for nanobind
brew install zlib
brew install jemalloc

# Use the python version that nanobind depends on
py_version=$(brew info nanobind --json | \
    jq -cr '.[].dependencies[] | select(. | startswith("python"))')
echo "Using python $py_version"
py_prefix=$(brew --prefix "$py_version") || exit 1

# numpy from brew pulls in a second python that fails to link, so install it
# in a venv based on the nanobind python instead
"$py_prefix/bin/python3" -m venv "$HOME/openvdb-venv" || exit 1
"$HOME/openvdb-venv/bin/pip" install numpy || exit 1

# Point CMake's FindPython at the venv so the python tests see numpy.
# export for subsequent action steps (note, not exported for this env)
echo "VIRTUAL_ENV=$HOME/openvdb-venv" >> $GITHUB_ENV
echo "$HOME/openvdb-venv/bin" >> $GITHUB_PATH

# use bash
echo "/usr/local/opt/bash/bin" >> $GITHUB_PATH
echo "/opt/homebrew/opt/bash/bin" >> $GITHUB_PATH

# use gnu-getopt
echo "/usr/local/opt/gnu-getopt/bin" >> $GITHUB_PATH
echo "/opt/homebrew/opt/gnu-getopt/bin" >> $GITHUB_PATH

LLVM_VERSION=$1
if [ ! -z "$LLVM_VERSION" ]; then
    if [ "$LLVM_VERSION" == "latest" ]; then
        brew install llvm
    else
        brew install llvm@$LLVM_VERSION
    fi
fi
