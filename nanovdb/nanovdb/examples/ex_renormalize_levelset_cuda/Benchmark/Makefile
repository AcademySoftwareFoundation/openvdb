CUDA_ROOT = $(shell dirname `which nvcc`)/..

NVCC=nvcc
NVCC_FLAGS=--forward-unknown-to-host-compiler --extended-lambda --std=c++17 -lineinfo --gpu-architecture=native -Xptxas -uumn
CUDA_LIBRARY=-L$(CUDA_ROOT)/lib64
CUDA_LIBS=-lcudart -lcuda

CC = g++
ABI_FLAG =
CFLAGS = -march=native -std=c++17 -Wall -w
INCLUDES = -I./include -I$(CUDA_ROOT)/include -I$(NANOVDB_ROOT) -I$(OPENVDB_ROOT)/include -DNANOVDB_USE_OPENVDB -DNANOVDB_USE_TBB
LIBRARY = $(CUDA_LIBRARY) -L$(OPENVDB_ROOT)/lib -L$(OPENVDB_ROOT)/lib64
LIBS = $(CUDA_LIBS) -lopenvdb -ltbb -ltbbmalloc_proxy -ltbbmalloc  -lpthread -lm
DEP = Makefile include/*.h include/*.cuh


BUILD_DEBUG   = $(NVCC) $(NVCC_FLAGS) $(CFLAGS) $(ABI_FLAG) -O0 -g
BUILD_RELEASE = $(NVCC) $(NVCC_FLAGS) $(CFLAGS) $(ABI_FLAG) -DNDEBUG -O3

# Avoid associating these build target with files of the same name
.PHONY: all clean 

all: release debug archive

def: release

.DEFAULT_GOAL := def

#########

release/main.o: $(DEP) src/main.cpp
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) -o release/main.o -c src/main.cpp

release/Benchmark.cu.o: $(DEP) src/Benchmark.cu
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) -o release/Benchmark.cu.o -c src/Benchmark.cu

release/BenchmarkIO.cpp.o: $(DEP) src/BenchmarkIO.cpp
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) -o release/BenchmarkIO.cpp.o -c src/BenchmarkIO.cpp

release/Benchmark.cpp.o: $(DEP) src/Benchmark.cpp
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) -o release/Benchmark.cpp.o -c src/Benchmark.cpp

release: release/main.o release/Benchmark.cu.o release/Benchmark.cpp.o release/BenchmarkIO.cpp.o
	$(BUILD_RELEASE) $(INCLUDES) $(LIBRARY) -o release/vdb_benchmark release/main.o release/Benchmark.cu.o release/Benchmark.cpp.o release/BenchmarkIO.cpp.o $(LIBS)

#########

debug/main.o: $(DEP) src/main.cpp
	@mkdir -p debug
	$(BUILD_DEBUG) $(INCLUDES) -o debug/main.o -c src/main.cpp

debug/BenchmarkIO.cpp.o: $(DEP) src/BenchmarkIO.cpp
	@mkdir -p debug
	$(BUILD_DEBUG) $(INCLUDES) -o debug/BenchmarkIO.cpp.o -c src/BenchmarkIO.cpp

debug/Benchmark.cpp.o: $(DEP) src/Benchmark.cpp
	@mkdir -p debug
	$(BUILD_DEBUG) $(INCLUDES) -o debug/Benchmark.cpp.o -c src/Benchmark.cpp

debug: debug/main.o debug/Benchmark.cpp.o debug/BenchmarkIO.cpp.o
	$(BUILD_DEBUG)  $(INCLUDES) $(LIBRARY) -o debug/vdb_benchmark debug/main.o debug/Benchmark.cpp.o debug/BenchmarkIO.cpp.o $(LIBS)

#########

archive:
	@tar -C .. -czf vdb_benchmark.tgz vdb_benchmark/Makefile vdb_benchmark/src vdb_benchmark/include vdb_benchmark/README.md

#########

clean:
	@rm -rf debug release vdb_benchmark.tgz
