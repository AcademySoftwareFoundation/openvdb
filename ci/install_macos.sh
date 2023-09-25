#!/usr/bin/env bash

set -x

brew update

if [ ! -z $2 ]; then
    if [[ $2 == "gcc"* || $2 == "llvm"* ]]; then
        brew install $2
    else
        # don't silently succeed
        echo "Unknown compiler type/version for second argument to install_macos.sh: $2"
        exit -1
    fi
fi

brew install bash gnu-getopt # for CI scripts
brew install cmake
brew install boost
brew install pybind11 # also installs the dependent python version
brew install zlib
brew install glfw
brew install googletest
brew install c-blosc
brew install jq # for trivial parsing of brew json
brew install tbb

# Alias python version installed by pybind11 to path
py_version=$(brew info pybind11 --json | \
    jq -cr '.[].dependencies[] | select(. | startswith("python"))')
echo "Using python $py_version"
# export for subsequent action steps (note, not exported for this env)
echo "Python_ROOT_DIR=/usr/local/opt/$py_version" >> $GITHUB_ENV
echo "/usr/local/opt/$py_version/bin" >> $GITHUB_PATH

# use gnu-getopt
echo "/usr/local/opt/gnu-getopt/bin" >> $GITHUB_PATH

LATEST=$1
if [ "$LATEST" == "latest" ]; then
    brew install openexr
else
    brew install ilmbase
    brew install openexr@2

    # Export OpenEXR paths which are no longer installed to /usr/local (as v2.x is deprecated)
    echo "IlmBase_ROOT=/usr/local/opt/ilmbase" >> $GITHUB_ENV
    echo "OpenEXR_ROOT=/usr/local/opt/openexr@2" >> $GITHUB_ENV
    echo "/usr/local/opt/openexr@2/bin" >> $GITHUB_PATH
fi
