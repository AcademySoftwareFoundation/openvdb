#!/usr/bin/env bash

# Download and install deps from homebrew on macos

brew update
brew install bash gnu-getopt # for CI scripts
brew install cmake
brew install boost
brew install cppunit
brew install c-blosc
brew install zlib

# use gnu-getopt
echo "/usr/local/opt/gnu-getopt/bin" >> $GITHUB_PATH

LLVM_VERSION=$1
if [ "$LLVM_VERSION" == "latest" ]; then
    brew install tbb
    brew install llvm
else
    brew install tbb@2020
    brew install llvm@$LLVM_VERSION

    # Export TBB paths which are no longer installed to /usr/local (as v2020 is deprecated)
    echo "TBB_ROOT=/usr/local/opt/tbb@2020" >> $GITHUB_ENV
    echo "/usr/local/opt/tbb@2020/bin" >> $GITHUB_PATH
fi
