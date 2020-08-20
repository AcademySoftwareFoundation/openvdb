#!/usr/bin/env bash

set -ex

TEST_NANOVDB_RESULTS=testNanoVDB.xml

if [ -d "__build_core/unittest" ]; then
	cd __build_core/unittest/Release
	mkdir -p data
	export GTEST_OUTPUT=xml:${TEST_NANOVDB_RESULTS}
	testNanoVDB
fi
