# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0

#  Build targets:
#
#  all: builds everything but runs nothing
#
#  test: build and run unit-tests
#
#  examples: build and run examples
#
#  cmd:  release build command-line tools
#
#  benchmark: build and run benchmark tests
#
#  clean: clean up the build directory

USE_OPENVDB := yes
USE_ZLIB   := yes
USE_BLOSC := yes
USE_CUDA := yes
USE_TBB := yes
USE_JEMALLOC := no
USE_OPENGL := yes
USE_OPENCL := yes
USE_INTRINSICS := yes

BUILD_BENCHMARK := yes
BUILD_UNITTESTS := yes
USE_GLFW := yes

ifeq (yes,$(USE_OPENVDB))
	USE_TBB = yes
	USE_ZLIB = yes
	USE_BLOSC = yes
endif

ifeq ("$(shell uname -s)","Darwin")
    USE_CUDA = no
    SHELL = /bin/tcsh
endif

#CC = clang -Wall -Wunused-parameter -march=native
CC = g++ -Wfatal-errors -Wall -Wunused-parameter -Wno-invalid-offsetof -march=native
#CC = g++ -Wall -Wunused-parameter -Wno-invalid-offsetof -march=native

ifeq (yes,$(USE_INTRINSICS))
    CXXFLAGS += -DNANOVDB_USE_INTRINSICS
endif 

ifeq (yes,$(USE_OPENVDB))
    CXXFLAGS += -std=c++14
else
    CXXFLAGS += -std=c++11
endif
BUILD_DEBUG   = $(CC) $(CXXFLAGS) -O1 -g
BUILD_RELEASE = $(CC) $(CXXFLAGS) -DNDEBUG -O3
INCLUDES = -I$(HOME)/local/include -I./ -Iexternal
LIBRARY = -L/usr/lib
LIBS = -lpthread
BRANCH = $(shell git rev-parse --abbrev-ref HEAD)
ARCHIVE = NanoVDB_$(shell date +%F_%Hh%Mm%Ss)_$(BRANCH)

DEP = Makefile nanovdb/*.h nanovdb/util/*.h
CMD = release/nanovdb_print release/nanovdb_convert release/nanovdb_viewer release/nanovdb_validate
EXA = debug/bump_pool_buffer debug/make_custom_nanovdb debug/make_nanovdb_sphere debug/make_typed_grids \
      debug/map_pool_buffer debug/read_nanovdb_sphere debug/read_nanovdb_sphere_accessor
ALL = $(CMD) $(EXA)

ifneq ($(wildcard $(HOME)/local),)
    LIBRARY += -L$(HOME)/local/lib -L$(HOME)/local/lib64
endif

ifeq ("$(shell uname -s)","Darwin")
ifeq (yes,$(USE_CUDA))
$(warning CUDA not supported on Darwin)
endif
	USE_CUDA = no
endif

ifeq (yes,$(USE_GLFW))
	USE_OPENGL = yes
	VIEWER_SRC_FILES += release/Viewer.o
	VIEWER_LIBS += -lglfw
endif

# tbb...
ifeq (yes,$(USE_JEMALLOC))
	LIBS += -ljemalloc
endif

# tbb...
ifeq (yes,$(USE_TBB))
	CXXFLAGS += -DNANOVDB_USE_TBB
	LIBS += -ltbb
endif

# zlib...
ifeq (yes,$(USE_ZLIB))
    CXXFLAGS += -DNANOVDB_USE_ZIP
	LIBS += -lz
endif

# blosc...
ifeq (yes,$(USE_BLOSC))
    CXXFLAGS += -DNANOVDB_USE_BLOSC
	LIBS += -lblosc
endif

# openvdb...
ifeq (yes,$(USE_OPENVDB))
	ifneq ("$(wildcard $(OPENVDB_INCLUDE_PATH))", "")
		INCLUDES += -I$(OPENVDB_INCLUDE_PATH)
	endif
	ifneq ("$(wildcard $(OPENVDB_LIBRARY_PATH))", "")
		LIBRARY += -L$(OPENVDB_LIBRARY_PATH)
	endif
	#CXXFLAGS += -DOPENVDB_ABI_VERSION_NUMBER=6
	CXXFLAGS += -DNANOVDB_USE_OPENVDB
	LIBS += -lopenvdb -lHalf
	EXA  += debug/openvdb_to_nanovdb debug/openvdb_to_nanovdb_accessor
ifneq ("$(wildcard $(USE_CUDA))", "")
    EXA += cuda/openvdb_to_nanovdb_cuda + cuda/modify_nanovdb_thrust
endif
endif

# OpenGL
ifeq (yes,$(USE_OPENGL))
ifeq ("$(shell uname -s)","Darwin")
	LIBS += -framework OpenGL
else
	LIBS += -lGL -ldl
endif
	CXXFLAGS += -DNANOVDB_USE_GLAD
	INCLUDES += -Iexternal/glad/include
	CXXFLAGS += -DNANOVDB_USE_OPENGL
	VIEWER_SRC_FILES += release/FrameBufferGL.o release/RenderLauncherGL.o release/glad.o
endif

# OpenCL
ifeq (yes,$(USE_OPENCL))
ifeq ("$(shell uname -s)","Darwin")
	LIBS += -framework OpenCL
else
	LIBS += -lOpenCL
endif
	CXXFLAGS += -DNANOVDB_USE_OPENCL
	VIEWER_SRC_FILES += release/RenderLauncherCL.o
endif

# cuda/opencl
ifeq (yes,$(USE_CUDA))

	CUDA_PATH ?= /usr/local/cuda
	NVCC = $(CUDA_PATH)/bin/nvcc
	BUILD_CUDA = $(NVCC) $(CXXFLAGS) -DNDEBUG -O3 -use_fast_math --extended-lambda
#	BUILD_CUDA = $(NVCC) $(CXXFLAGS) -O1 -g -use_fast_math

ifneq ("$(wildcard $(CUDA_PATH))", "")
	LIBRARY += -L$(CUDA_PATH)/lib64
	INCLUDES += -I$(CUDA_PATH)/include
	#LIBS += -lcudart_static
endif
	CXXFLAGS += -DNANOVDB_USE_CUDA
	EXA += cuda/read_nanovdb_sphere_accessor_cuda
	VIEWER_SRC_FILES += release/RenderLauncherCUDA.o

else
	BUILD_CUDA = $(BUILD_RELEASE)
endif

# build unit-tests...
ifeq (yes,$(BUILD_UNITTESTS))
    LIBS += -lgtest -lpthread

	TST += test_nano
ifeq (yes,$(USE_OPENVDB))
	TST  += test_open
endif

	ALL += release/testNanoVDB release/testOpenVDB
endif

# build benchmark...
ifeq (yes,$(BUILD_BENCHMARK))
    LIBS += -lgtest -lpthread
	ALL += cuda/benchmark
endif

# build interactive-renderer...
ifeq (yes,$(USE_GLFW))
	CXXFLAGS += -DNANOVDB_USE_GLFW
endif

# Avoid associating these build target with files of the same name
.PHONY: all run clean test archive

.DEFAULT_GOAL := release


# build targets

all: $(ALL)

test: $(TST)

examples: $(EXA)

run_examples: $(EXA)
	$(EXA)

cmd: $(CMD)

ifeq (yes, $(BUILD_UNITTESTS))
test_nano: release/testNanoVDB
	./release/testNanoVDB --gtest_break_on_failure --gtest_filter="*" --gtest_repeat=1

test_open: release/testOpenVDB
	./release/testOpenVDB --gtest_break_on_failure --gtest_filter="*" --gtest_repeat=1

test_debug: debug/testNanoVDB debug/testOpenVDB
	./debug/testNanoVDB --gtest_break_on_failure --gtest_filter="*:-*CNanoVDB:*MultiFile2" --gtest_repeat=1
	./debug/testOpenVDB --gtest_break_on_failure --gtest_filter="*:-*CNanoVDB:*MultiFile2" --gtest_repeat=1
endif

ifeq (yes, $(USE_OPENVDB))
data/test.nvdb: release/testOpenVDB
	./release/testOpenVDB --gtest_filter="*OpenToNanoVDB" --gtest_repeat=1

benchmark: data/test.nvdb cuda/benchmark
	./cuda/benchmark --gtest_filter="*" --gtest_repeat=1
else
benchmark: cuda/benchmark
	./cuda/benchmark --gtest_filter="*" --gtest_repeat=1
endif

# Examples

bump_pool_buffer: debug/bump_pool_buffer
	debug/bump_pool_buffer

make_custom_nanovdb: debug/make_custom_nanovdb
	debug/make_custom_nanovdb

make_nanovdb_sphere: debug/make_nanovdb_sphere
	debug/make_nanovdb_sphere

make_typed_grids: debug/make_typed_grids
	debug/make_typed_grids

map_pool_buffer: debug/map_pool_buffer
	debug/map_pool_buffer

openvdb_to_nanovdb: debug/openvdb_to_nanovdb
	debug/openvdb_to_nanovdb

openvdb_to_nanovdb_cuda: cuda/openvdb_to_nanovdb_cuda
	cuda/openvdb_to_nanovdb_cuda

modify_nanovdb_thrust: cuda/modify_nanovdb_thrust
	cuda/modify_nanovdb_thrust

openvdb_to_nanovdb_accessor: debug/openvdb_to_nanovdb_accessor
	debug/openvdb_to_nanovdb_accessor

openvdb_to_nanovdb_accessor_cuda: cuda/openvdb_to_nanovdb_accessor_cuda
	cuda/openvdb_to_nanovdb_accessor_cuda

read_nanovdb_sphere: debug/read_nanovdb_sphere
	debug/read_nanovdb_sphere

read_nanovdb_sphere_accessor: debug/read_nanovdb_sphere_accessor
	debug/read_nanovdb_sphere_accessor

read_nanovdb_sphere_accessor_cuda: cuda/read_nanovdb_sphere_accessor_cuda
	cuda/read_nanovdb_sphere_accessor_cuda

write_nanovdb_grids: debug/write_nanovdb_grids
	debug/write_nanovdb_grids

# Command-line tools

nanovdb_print: release/nanovdb_print

ifeq (yes, $(USE_OPENVDB))
nanovdb_convert: release/nanovdb_convert
else
nanovdb_convert:
	$(warning WARNING: nanovdb_convert requires OpenVDB. Set USE_OPENVDB := yes)
endif

nanovdb_viewer: release/nanovdb_viewer

nanovdb_validate: release/nanovdb_validate

# build object files

release/TestNanoVDB.o: unittest/TestNanoVDB.cpp $(DEP)
	@mkdir -p release data
	$(BUILD_RELEASE) $(INCLUDES) -o $@ -c $<

debug/TestNanoVDB.o: unittest/TestNanoVDB.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

release/TestOpenVDB.o: unittest/TestOpenVDB.cpp $(DEP)
	@mkdir -p release data
	$(BUILD_RELEASE) $(INCLUDES) -o $@ -c $<

debug/TestOpenVDB.o: unittest/TestOpenVDB.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

cuda/TestNanoVDB.o: unittest/TestNanoVDB.cpp $(DEP)
	@mkdir -p cuda data
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

cuda/Benchmark.o: examples/benchmark/Benchmark.cpp examples/benchmark/*.h $(DEP)
	@mkdir -p cuda data
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

cuda/BenchKernels.o: examples/benchmark/BenchKernels.cu examples/benchmark/*.h $(DEP)
	@mkdir -p cuda data
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

# Examples

debug/bump_pool_buffer.o: examples/ex_bump_pool_buffer/bump_pool_buffer.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

debug/make_custom_nanovdb.o: examples/ex_make_custom_nanovdb/make_custom_nanovdb.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

debug/make_nanovdb_sphere.o: examples/ex_make_nanovdb_sphere/make_nanovdb_sphere.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

debug/make_typed_grids.o: examples/ex_make_typed_grids/make_typed_grids.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $< $(LIBS)

debug/map_pool_buffer.o: examples/ex_map_pool_buffer/map_pool_buffer.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

debug/openvdb_to_nanovdb.o: examples/ex_openvdb_to_nanovdb/openvdb_to_nanovdb.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

debug/openvdb_to_nanovdb_accessor.o: examples/ex_openvdb_to_nanovdb_accessor/openvdb_to_nanovdb_accessor.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

cuda/openvdb_to_nanovdb_cuda.o: examples/ex_openvdb_to_nanovdb_cuda/openvdb_to_nanovdb_cuda.cpp $(DEP)
	@mkdir -p cuda data
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

cuda/openvdb_to_nanovdb_cuda_kernel.o: examples/ex_openvdb_to_nanovdb_cuda/openvdb_to_nanovdb_cuda.cu $(DEP)
	@mkdir -p cuda data
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

cuda/modify_nanovdb_thrust.o: examples/ex_modify_nanovdb_thrust/modify_nanovdb_thrust.cu $(DEP)
	@mkdir -p cuda data
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

debug/read_nanovdb_sphere.o: examples/ex_read_nanovdb_sphere/read_nanovdb_sphere.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

debug/read_nanovdb_sphere_accessor.o: examples/ex_read_nanovdb_sphere_accessor/read_nanovdb_sphere_accessor.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

cuda/read_nanovdb_sphere_accessor_cuda.o: examples/ex_read_nanovdb_sphere_accessor_cuda/read_nanovdb_sphere_accessor_cuda.cpp $(DEP)
	@mkdir -p cuda data
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

cuda/read_nanovdb_sphere_accessor_cuda_kernel.o: examples/ex_read_nanovdb_sphere_accessor_cuda/read_nanovdb_sphere_accessor_cuda.cu $(DEP)
	@mkdir -p cuda data
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

debug/write_nanovdb_grids.o: examples/ex_write_nanovdb_grids/write_nanovdb_grids.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

debug/vox_to_nanovdb.o: examples/ex_vox_to_nanovdb/vox_to_nanovdb.cpp $(DEP)
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o $@ -c $<

# Command-line tools

release/nanovdb_print.o: cmd/print/nanovdb_print.cpp $(DEP)
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) -o $@ -c $<

release/nanovdb_convert.o: cmd/convert/nanovdb_convert.cpp $(DEP)
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) -o $@ -c $<

release/nanovdb_validate.o: cmd/validate/nanovdb_validate.cpp $(DEP)
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) -o $@ -c $<

release/nanovdb_viewer.o: cmd/viewer/nanovdb_viewer.cpp $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/Renderer.o: cmd/viewer/Renderer.cpp cmd/viewer/Renderer.h cmd/viewer/RenderLauncher.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/Viewer.o: cmd/viewer/Viewer.cpp cmd/viewer/Viewer.h cmd/viewer/RenderLauncher.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/BatchRenderer.o: cmd/viewer/BatchRenderer.cpp cmd/viewer/BatchRenderer.h cmd/viewer/RenderLauncher.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/FrameBuffer.o: cmd/viewer/FrameBuffer.cpp cmd/viewer/FrameBuffer.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/FrameBufferGL.o: cmd/viewer/FrameBufferGL.cpp cmd/viewer/FrameBufferGL.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/FrameBufferHost.o: cmd/viewer/FrameBufferHost.cpp cmd/viewer/FrameBufferHost.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/RenderLauncher.o: cmd/viewer/RenderLauncher.cpp cmd/viewer/RenderUtils.h cmd/viewer/RenderPointsUtils.h cmd/viewer/RenderGridUtils.h cmd/viewer/RenderLevelSetUtils.h cmd/viewer/RenderFogVolumeUtils.h cmd/viewer/RenderVoxelUtils.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/RenderLauncherC99impl.o: cmd/viewer/RenderLauncherC99impl.c $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/RenderLauncherC99.o: cmd/viewer/RenderLauncherC99.cpp $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/RenderLauncherCL.o: cmd/viewer/RenderLauncherCL.cpp $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/RenderLauncherCUDA.o: cmd/viewer/RenderLauncherCUDA.cu cmd/viewer/RenderUtils.h cmd/viewer/RenderPointsUtils.h cmd/viewer/RenderGridUtils.h cmd/viewer/RenderLevelSetUtils.h cmd/viewer/RenderFogVolumeUtils.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/RenderLauncherGL.o: cmd/viewer/RenderLauncherGL.cpp $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/primitives.o: cmd/viewer/primitives.cpp $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/RenderLauncherCpuMT.o: cmd/viewer/RenderLauncherCpuMT.cpp cmd/viewer/RenderUtils.h cmd/viewer/RenderPointsUtils.h cmd/viewer/RenderGridUtils.h cmd/viewer/RenderLevelSetUtils.h cmd/viewer/RenderFogVolumeUtils.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/glad.o: external/glad/src/glad.c
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/AssetLoader.o: cmd/viewer/AssetLoader.cpp cmd/viewer/AssetLoader.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/CallbackPool.o: cmd/viewer/CallbackPool.cpp cmd/viewer/CallbackPool.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/GridAssetUrl.o: cmd/viewer/GridAssetUrl.cpp cmd/viewer/GridAssetUrl.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

release/GridManager.o: cmd/viewer/GridManager.cpp cmd/viewer/GridManager.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -Iexamples/ex_vox_to_nanovdb -o $@ -c $<

release/StringUtils.o: cmd/viewer/StringUtils.cpp cmd/viewer/StringUtils.h $(DEP)
	@mkdir -p release
	$(BUILD_CUDA) $(INCLUDES) -o $@ -c $<

# build executables

release/testNanoVDB: release/TestNanoVDB.o
	$(BUILD_RELEASE) $(LIBRARY) -o $@ $^ $(LIBS)

debug/testNanoVDB: debug/TestNanoVDB.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

ifeq (yes,$(USE_OPENVDB))
release/testOpenVDB: release/TestOpenVDB.o
	$(BUILD_RELEASE) $(LIBRARY) -o $@ $^ $(LIBS)

debug/testOpenVDB: debug/TestOpenVDB.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)
else
release/testOpenVDB:
	$(warning WARNING: testOpenVDB requires OpenVDB. Set USE_OPENVDB := yes)

debug/testOpenVDB:
	$(warning WARNING: testOpenVDB requires OpenVDB. Set USE_OPENVDB := yes)
endif

cuda/benchmark: cuda/Benchmark.o cuda/BenchKernels.o
	$(BUILD_CUDA) $(LIBRARY) -o $@ $^ $(LIBS)

debug/bump_pool_buffer: debug/bump_pool_buffer.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

debug/make_custom_nanovdb: debug/make_custom_nanovdb.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

debug/make_nanovdb_sphere: debug/make_nanovdb_sphere.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

debug/write_nanovdb_grids: debug/write_nanovdb_grids.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

debug/make_typed_grids: debug/make_typed_grids.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

debug/map_pool_buffer: debug/map_pool_buffer.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

debug/vox_to_nanovdb: debug/vox_to_nanovdb.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

ifeq (yes,$(USE_OPENVDB))
debug/openvdb_to_nanovdb: debug/openvdb_to_nanovdb.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

cuda/openvdb_to_nanovdb_cuda: cuda/openvdb_to_nanovdb_cuda.o cuda/openvdb_to_nanovdb_cuda_kernel.o
	$(BUILD_CUDA) $(LIBRARY) -o $@ $^ $(LIBS)

cuda/modify_nanovdb_thrust: cuda/modify_nanovdb_thrust.o
	$(BUILD_CUDA) $(LIBRARY) -o $@ $^ $(LIBS)

debug/openvdb_to_nanovdb_accessor: debug/openvdb_to_nanovdb_accessor.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

cuda/openvdb_to_nanovdb_accessor_cuda: cuda/openvdb_to_nanovdb_accessor_cuda.o cuda/openvdb_to_nanovdb_cuda_kernel.o
	$(BUILD_CUDA) $(LIBRARY) -o $@ $^ $(LIBS)
else
	$(warning WARNING: openvdb_to_nanovdb requires OpenVDB. Set USE_OPENVDB := yes)
endif

debug/read_nanovdb_sphere: debug/read_nanovdb_sphere.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

debug/read_nanovdb_sphere_accessor: debug/read_nanovdb_sphere_accessor.o
	$(BUILD_DEBUG) $(LIBRARY) -o $@ $^ $(LIBS)

cuda/read_nanovdb_sphere_accessor_cuda: cuda/read_nanovdb_sphere_accessor_cuda.o cuda/read_nanovdb_sphere_accessor_cuda_kernel.o
	$(BUILD_CUDA) $(LIBRARY) -o $@ $^  $(LIBS)

# command-line tools

release/nanovdb_print: release/nanovdb_print.o
	$(BUILD_RELEASE) $(LIBRARY) -o $@ $^ $(LIBS)

ifeq (yes, $(USE_OPENVDB))
release/nanovdb_convert: release/nanovdb_convert.o
	$(BUILD_RELEASE) $(LIBRARY) -o $@ $^ $(LIBS)
else
release/nanovdb_convert:
	$(warning WARNING: nanovdb_convert requires OpenVDB. Set USE_OPENVDB := yes)
endif

release/nanovdb_validate: release/nanovdb_validate.o
	$(BUILD_RELEASE) $(LIBRARY) -o $@ $^ $(LIBS)

release/nanovdb_viewer: release/nanovdb_viewer.o \
                        release/FrameBuffer.o \
                        release/FrameBufferHost.o \
						release/RenderLauncher.o \
						release/RenderLauncherC99impl.o \
						release/RenderLauncherC99.o \
						release/RenderLauncherCpuMT.o \
						release/Renderer.o \
						release/BatchRenderer.o \
						release/AssetLoader.o \
						release/CallbackPool.o \
						release/GridAssetUrl.o \
						release/GridManager.o \
						release/StringUtils.o \
						release/primitives.o \
						$(VIEWER_SRC_FILES)
	$(BUILD_CUDA) $(LIBRARY) -o $@ $^ $(VIEWER_LIBS) $(LIBS)

# build other

movie:
	@ffmpeg -y -framerate 30 -i $(VDB_SCRATCH_PATH)/nanovdb_gpu_%03d.ppm $(VDB_SCRATCH_PATH)/nanovdb_30fps_v0.mp4

dos2unix:
	@find . -type f -print0 | xargs -0 dos2unix -q

archive:
	@mkdir -p archive
	@git archive $(BRANCH) | gzip > archive/$(ARCHIVE).tgz

clean:
	@rm -rf archive build cuda debug release

clean-all: clean
	@rm -rf data *.tgz *.vdb *.nvdb imgui.ini
