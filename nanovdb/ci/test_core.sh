#!/usr/bin/env bash

set -ex

NAME="$1"; shift

export VDB_DATA_PATH=/repo/vdbs
if [ ! -d ${VDB_DATA_PATH} ]; then
	./ci/download_vdbs.sh
fi

if [ -f "__build_core/${NAME}/unittest/testNanoVDB" ]; then

	TEST_NANOVDB_RESULTS=test-results/testNanoVDB/${NAME}.xml

	mkdir -p test-results/testNanoVDB
	pushd __build_core/${NAME}/unittest
	mkdir -p data
	chmod +x testNanoVDB
	export GTEST_OUTPUT=xml:../../../${TEST_NANOVDB_RESULTS}
	./testNanoVDB
	popd
fi

if [ -f "__build_core/${NAME}/unittest/testOpenVDB" ]; then

	TEST_NANOVDB_RESULTS=test-results/testOpenVDB/${NAME}.xml

	mkdir -p test-results/testOpenVDB
	pushd __build_core/${NAME}/unittest
	mkdir -p data
	chmod +x testOpenVDB
	export GTEST_OUTPUT=xml:../../../${TEST_NANOVDB_RESULTS}
	./testOpenVDB
	popd
fi
