#!/usr/bin/env bash

#if [ $# -eq 0 ]; then
#	echo "Usage: test.sh <branch|commit>"
#	exit 1
#fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
GIT_BRANCH=$1
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
    docker build -t ${IMAGE}:dev-test -f ${DOCKER_DIR}/Dockerfile.test ${DOCKER_DIR}
fi