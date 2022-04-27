#!/usr/bin/env bash

set -e

# print versions
bash --version
if [ ! -z "$CXX" ]; then $CXX -v; fi
cmake --version
CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)

################################################

OPTS_ARGS+=("t:")  ## See --target
OPTS_ARGS+=("j:")  ## Thread usage
OPTS_ARGS+=("c:")  ## See --cargs
OPTS_ARGS+=("v")   ## See --verbose
OPTL_ARGS+=("components:")  ## Specify cmake component(s) to enable
OPTL_ARGS+=("config:")      ## Specify cmake configuration during the build step
OPTL_ARGS+=("target:")      ## Specify target(s) to build
OPTL_ARGS+=("cargs:")       ## args to pass directly to cmake generation step
OPTL_ARGS+=("build-type:")  ## Release, Debug, etc.
OPTL_ARGS+=("verbose")      ## Verbose build output

# Defaults
declare -A PARMS
PARMS[--components]=core,bin
PARMS[--target]=install
# github actions runners have 2 threads
# https://help.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners
PARMS[-j]=2

# Available options for --components
declare -A COMPONENTS
COMPONENTS['core']='OPENVDB_BUILD_CORE'
COMPONENTS['python']='OPENVDB_BUILD_PYTHON_MODULE'
COMPONENTS['test']='OPENVDB_BUILD_UNITTESTS'
COMPONENTS['bin']='OPENVDB_BUILD_BINARIES'
COMPONENTS['view']='OPENVDB_BUILD_VDB_VIEW'
COMPONENTS['render']='OPENVDB_BUILD_VDB_RENDER'
COMPONENTS['hou']='OPENVDB_BUILD_HOUDINI_PLUGIN'
COMPONENTS['doc']='OPENVDB_BUILD_DOCS'

COMPONENTS['axcore']='OPENVDB_BUILD_AX'
COMPONENTS['axgr']='OPENVDB_BUILD_AX_GRAMMAR'
COMPONENTS['axbin']='OPENVDB_BUILD_AX_BINARIES'
COMPONENTS['axtest']='OPENVDB_BUILD_AX_UNITTESTS'

COMPONENTS['nano']='OPENVDB_BUILD_NANOVDB'
COMPONENTS['nanotest']='NANOVDB_BUILD_UNITTESTS'
COMPONENTS['nanoexam']='NANOVDB_BUILD_EXAMPLES'
COMPONENTS['nanobench']='NANOVDB_BUILD_BENCHMARK'
COMPONENTS['nanotool']='NANOVDB_BUILD_TOOLS'

################################################

HAS_PARM() {
    if [ -z "${PARMS[$1]}" ]; then return 1
    else return 0; fi
}

# Format to string and replace spaces with commas
LARGS_STR="${OPTL_ARGS[@]}"
LARGS_STR=${LARGS_STR// /,}
SARGS_STR="${OPTS_ARGS[@]}"
SARGS_STR=${SARGS_STR// /,}

# Parse all arguments and store them in an array, split by whitespace. Error if unsupported
ARGS="$(eval getopt --options=$SARGS_STR --longoptions=$LARGS_STR -- "$@")"
eval set -- "$ARGS"

# split into associative array
while true; do
    case "$1" in
        -v|--verbose) # options which dont take an argument
            PARMS["$1"]="ON"; shift
            ;;
        -[a-z]*|--[a-z]*) # all other arguments (key/values)
            PARMS["$1"]="$2"; shift 2
            ;;
        --)
            shift; break
            ;;
    esac
done

################################################

# extract arguments
if HAS_PARM -t; then TARGET=${PARMS[-t]}; fi
if HAS_PARM --target; then
    if [ -z $TARGET ]; then TARGET=${PARMS[--target]}
    else TARGET+=","${PARMS[--target]}; fi
fi
if HAS_PARM -c; then CMAKE_EXTRA=${PARMS[-c]}; fi
if HAS_PARM --cargs; then
    if [ -z $CMAKE_EXTRA ]; then CMAKE_EXTRA=${PARMS[--cargs]}
    else CMAKE_EXTRA+=" "${PARMS[--cargs]}; fi
fi
# handle whitespace
eval "CMAKE_EXTRA=($CMAKE_EXTRA)"

if HAS_PARM -v || HAS_PARM --verbose; then
    # Using CMAKE_VERBOSE_MAKEFILE as well as `cmake --verbose` to
    # support older versions of CMake.
    CMAKE_EXTRA+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
fi
if HAS_PARM --build-type; then CMAKE_EXTRA+=("-DCMAKE_BUILD_TYPE=${PARMS[--build-type]}"); fi

# Available components. If a component is not provided it is
# explicitly set to OFF.
IN_COMPONENTS=${PARMS[--components]}
IFS=', ' read -r -a IN_COMPONENTS <<< "$IN_COMPONENTS"
for comp in "${IN_COMPONENTS[@]}"; do
    if [ -z ${COMPONENTS[$comp]} ]; then
        echo "Invalid component passed to build \"$comp\""; exit -1
    fi
done
# Build Components command
for comp in "${!COMPONENTS[@]}"; do
    setting="OFF"
    for in in "${IN_COMPONENTS[@]}"; do
        if [[ $comp == "$in" ]]; then
            setting="ON"; break
        fi
    done
    CMAKE_EXTRA+=("-D${COMPONENTS[$comp]}=$setting")
done

################################################

# github actions runners have 2 threads
# https://help.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners
export CMAKE_BUILD_PARALLEL_LEVEL=${PARMS[-j]}
echo "Build using ${CMAKE_BUILD_PARALLEL_LEVEL} threads"

# NOTE: --parallel only effects the number of projects build, not t-units.
# We support this with out own MSVC_MP_THREAD_COUNT option for MSVC.
# Alternatively it is mentioned that the following should work:
#   cmake --build . --  /p:CL_MPcount=8
# However it does not seem to for our project.
# https://gitlab.kitware.com/cmake/cmake/-/issues/20564

CMAKE_BUILD_CMD="cmake --build . --parallel ${PARMS[-j]} --target $TARGET --verbose"

if HAS_PARM --config; then
    CMAKE_BUILD_CMD+=" --config ${PARMS[--config]}"
fi

################################################

mkdir -p build
cd build

# Report the cmake commands
set -x

# Note:
# - print and lod binary options are always on and can be toggles with: OPENVDB_BUILD_BINARIES=ON/OFF
cmake \
    -DOPENVDB_USE_DEPRECATED_ABI_7=ON \
    -DOPENVDB_USE_DEPRECATED_ABI_8=ON \
    -DOPENVDB_BUILD_VDB_PRINT=ON \
    -DOPENVDB_BUILD_VDB_LOD=ON \
    -DMSVC_MP_THREAD_COUNT=${PARMS[-j]} \
    "${CMAKE_EXTRA[@]}" \
    ..

$CMAKE_BUILD_CMD
