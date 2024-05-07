#!/usr/bin/env bash

set -x

# Remove Python3 symlinks in /usr/local/bin as workaround to brew update issues
# https://github.com/actions/setup-python/issues/577
rm /usr/local/bin/2to3* || :
rm /usr/local/bin/idle3* || :
rm /usr/local/bin/pydoc* || :
rm /usr/local/bin/python3* || :

brew update
brew install bash gnu-getopt # for CI scripts
brew install boost
brew install c-blosc
brew install cmake
brew install glfw
brew install googletest
brew install jq # for trivial parsing of brew json
brew install openexr
brew install pybind11 # also installs the dependent python version
brew install tbb
brew install zlib
brew install jemalloc

# Alias python version installed by pybind11 to path
py_version=$(brew info pybind11 --json | \
    jq -cr '.[].dependencies[] | select(. | startswith("python"))')
echo "Using python $py_version"
# export for subsequent action steps (note, not exported for this env)
echo "Python_ROOT_DIR=/usr/local/opt/$py_version" >> $GITHUB_ENV
echo "/usr/local/opt/$py_version/bin" >> $GITHUB_PATH

# use gnu-getopt
echo "/usr/local/opt/gnu-getopt/bin" >> $GITHUB_PATH

LLVM_VERSION=$1
if [ ! -z "$LLVM_VERSION" ]; then
    if [ "$LLVM_VERSION" == "latest" ]; then
        brew install llvm
        brew install cppunit
    else
        brew install llvm@$LLVM_VERSION
        brew install cppunit
    fi
fi
