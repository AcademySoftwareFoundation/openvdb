#!/usr/bin/env bash

set -ex

NAME="$1"; shift
OUT_PATH=../../__output/${NAME}

if [ -d "__release_core/${NAME}/bin" ]; then
	cd __release_core/${NAME}	
	mkdir -p ${OUT_PATH}

	# make gold image.
	./bin/nanovdb_viewer -b --render-start 0 --render-end 0 --render-output ${OUT_PATH}/gold.%04d.png -p host-mt --render-camera-turntable 1 internal://#ls_box_100

	# test against other platforms...
if false; then
	./bin/nanovdb_viewer -b --render-end 0 --render-gold ${OUT_PATH}/gold.%04d.png --render-output ${OUT_PATH}/test-cuda.%04d.png -p cuda --render-camera-turntable 1 internal://#ls_box_100
	./bin/nanovdb_viewer -b --render-end 0 --render-gold ${OUT_PATH}/gold.%04d.png --render-output ${OUT_PATH}/test-c99.%04d.png -p host-c99 --render-camera-turntable 1 internal://#ls_box_100
	./bin/nanovdb_viewer -b --render-end 0 --render-gold ${OUT_PATH}/gold.%04d.png --render-output ${OUT_PATH}/test-glsl.%04d.png -p glsl --render-camera-turntable 1 internal://#ls_box_100
	./bin/nanovdb_viewer -b --render-end 0 --render-gold ${OUT_PATH}/gold.%04d.png --render-output ${OUT_PATH}/test-opencl.%04d.png -p opencl --render-camera-turntable 1 internal://#ls_box_100
	./bin/nanovdb_viewer -b --render-end 0 --render-gold ${OUT_PATH}/gold.%04d.png --render-output ${OUT_PATH}/test-optix.%04d.png -p optix --render-camera-turntable 1 internal://#ls_box_100
	
fi
fi
