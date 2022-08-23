#!/usr/bin/env bash

# Download and install deps from homebrew on macos

LLVM_VERSION=$1
if [ -z $LLVM_VERSION ]; then
    echo "No LLVM version provided for LLVM installation"
    exit -1
fi

brew update
brew install bash gnu-getopt # for CI scripts
brew install cmake
brew install boost
brew install boost-python3 # also installs the dependent python version
brew install cppunit
brew install c-blosc
brew install tbb@2020
brew install llvm@$LLVM_VERSION
brew install zlib
brew install jq # for trivial parsing of brew json

# Alias python version installed by boost-python3 to path
py_version=$(brew info boost-python3 --json | \
    jq -cr '.[].dependencies[] | select(. | startswith("python"))')
echo "Using python $py_version"
# export for subsequent action steps (note, not exported for this env)
echo "Python_ROOT_DIR=/usr/local/opt/$py_version" >> $GITHUB_ENV
echo "/usr/local/opt/$py_version/bin" >> $GITHUB_PATH

# Export TBB paths which are no longer installed to /usr/local (as v2020 is deprecated)
echo "TBB_ROOT=/usr/local/opt/tbb@2020" >> $GITHUB_ENV
echo "/usr/local/opt/tbb@2020/bin" >> $GITHUB_PATH

# use gnu-getopt
echo "/usr/local/opt/gnu-getopt/bin" >> $GITHUB_PATH
