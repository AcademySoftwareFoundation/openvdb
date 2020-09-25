#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DOCKER_DIR=${DIR}
DIST_DIR=${DOCKER_DIR}/__dist
REPO_DIR=${DOCKER_DIR}/../..
echo "DOCKER_DIR: ${DOCKER_DIR}"

IMAGE="oddsocks/nanovdb"

# -- copy local files to release directory...
mkdir -p ${DIST_DIR}/scripts
pushd ${DOCKER_DIR}/..

# tar -cvf ${DIST_DIR}/scripts.tar scripts
rm -rf ${DIST_DIR}/scripts/*
cp -R scripts/* ${DIST_DIR}/scripts/

popd

# -- build the image.
docker build -t ${IMAGE}:dev-base -f ${DOCKER_DIR}/Dockerfile.base ${DOCKER_DIR}
