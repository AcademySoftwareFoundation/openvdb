#!/usr/bin/env bash

set -ex

NAME="$1"; shift
OUT_PATH=../../__output/${NAME}

if [ -d "__release_core/${NAME}/bin" ]; then
	cd __release_core/${NAME}	
	mkdir -p ${OUT_PATH}

	# make gold image.
	./bin/nanovdb_viewer -b -o ${OUT_PATH}/gold -p host-mt --count 1 --turntable
	mogrify -format png ${OUT_PATH}/gold.0000.pfm

	# test against other platforms...

	./bin/nanovdb_viewer -b --gold ${OUT_PATH}/gold -o ${OUT_PATH}/test-cuda -p cuda -n 1 --turntable
	./bin/nanovdb_viewer -b --gold ${OUT_PATH}/gold -o ${OUT_PATH}/test-c99 -p host-c99 -n 1 --turntable
	./bin/nanovdb_viewer -b --gold ${OUT_PATH}/gold -o ${OUT_PATH}/test-glsl -p glsl -n 1 --turntable
	./bin/nanovdb_viewer -b --gold ${OUT_PATH}/gold -o ${OUT_PATH}/test-opencl -p opencl -n 1 --turntable
	./bin/nanovdb_viewer -b --gold ${OUT_PATH}/gold -o ${OUT_PATH}/test-optix -p optix -n 1 --turntable

	#compare -verbose -metric MAE ../../__output/gold.0000.pfm ../../__output/test-glsl.0000.pfm null: 2>&1
	#compare -verbose -metric MAE ../../__output/gold.0000.pfm ../../__output/test-opencl.0000.pfm null: 2>&1
fi
