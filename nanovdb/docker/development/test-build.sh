#!/usr/bin/env bash

#if [ $# -eq 0 ]; then
#	echo "Usage: test-build.sh <compiler e.g. g++,g++-8,clang++,etc.> <cuda_version=9.2,10.2> <branch|commit>"
#	exit 1
#fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
COMPILER_NAME=$1; shift
CUDA_VER=$1; shift

if [[ -z "${COMPILER_NAME}" ]]; then
    COMPILER_NAME="clang++"
else if [[ -z "${CUDA_VER}" ]]; then
    CUDA_VER="10.2"
else
    GIT_BRANCH=$1; shift
fi

echo "Building with compiler: ${COMPILER_NAME}"
echo "CUDA version: ${CUDA_VER}"

DOCKER_DIR=${DIR}
DIST_DIR=${DOCKER_DIR}/__dist
REPO_DIR=${DOCKER_DIR}/../..
echo "DOCKER_DIR: ${DOCKER_DIR}"

IMAGE="oddsocks/nanovdb"

# -- copy local files to release directory...
mkdir -p ${DIST_DIR}
if [[ ! -z "${GIT_BRANCH}" ]]; then
    pushd ${REPO_DIR}
    git archive ${GIT_BRANCH} -o ${DIST_DIR}/repo.tar
    popd
else
    pushd ${REPO_DIR}
    tar --exclude="data" --exclude="out" --exclude=".git" --exclude="__*" -cvf ${DIST_DIR}/repo.tar .
    popd
fi

# -- build the image.
if [ -f ${DIST_DIR}/repo.tar ]; then
    docker build -t ${IMAGE}:dev-test-build -f ${DOCKER_DIR}/Dockerfile.test-build --build-arg COMPILER=${COMPILER_NAME} --build-arg CUDA_VER=${CUDA_VER} ${DOCKER_DIR}
fi