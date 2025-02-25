# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(fvdb LANGUAGES CXX CUDA)

### Conda environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_ENV_PATH $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment path: ${CONDA_ENV_PATH}")
else()
    message(FATAL_ERROR "Conda environment path not found. Please activate the fvdb conda environment.")
endif()

### Get dependencies
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_cpm.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_nvtx.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_torch.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_tinyply.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_glm.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_cutlass.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_cudnn_frontend.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_blosc.cmake)

### Build options
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google & nvbench) benchmarks" OFF)

message(VERBOSE "FVDB: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE "FVDB: Configure CMake to build (google & nvbench) benchmarks: ${BUILD_BENCHMARKS}")

set(FVDB_BUILD_TESTS ${BUILD_TESTS})
set(FVDB_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})

### Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

### Source files
set(FVDB_CPP_FILES
    Config.cpp
    FVDB.cpp
    GaussianSplatting.cpp
    GridBatch.cpp
    JaggedTensor.cpp
    SparseConvPackInfo.cpp
    detail/autograd/Attention.cpp
    detail/autograd/AvgPoolGrid.cpp
    detail/autograd/GaussianRender.cpp
    detail/autograd/JaggedReduce.cpp
    detail/autograd/MaxPoolGrid.cpp
    detail/autograd/ReadIntoDense.cpp
    detail/autograd/SampleGrid.cpp
    detail/autograd/SparseConvolutionHalo.cpp
    detail/autograd/SplatIntoGrid.cpp
    detail/autograd/TransformPoints.cpp
    detail/autograd/UpsampleGrid.cpp
    detail/autograd/VolumeRender.cpp
    detail/build/CoarseFromFine.cpp
    detail/build/ConvGrid.cpp
    detail/build/DenseGrid.cpp
    detail/build/EmptyGrid.cpp
    detail/build/FineFromCoarse.cpp
    detail/build/FromMesh.cpp
    detail/build/NearestNeighborGridFromPoints.cpp
    detail/build/PaddedGridFromCoords.cpp
    detail/build/PaddedGridFromGrid.cpp
    detail/build/PaddedGridFromPoints.cpp
    detail/io/LoadNanovdb.cpp
    detail/io/SaveNanoVDB.cpp
    detail/TorchDeviceBuffer.cpp)

set(FVDB_CU_FILES
    detail/ops/MarchingCubes.cu
    detail/ops/ReadIntoDense.cu
    detail/ops/JOffsetsFromJIdx.cu
    detail/ops/TransformPointToGrid.cu
    detail/ops/CubesInGrid.cu
    detail/ops/JCat0.cu
    detail/ops/GridEdgeNetwork.cu
    detail/ops/SampleGridTrilinearWithGradBackward.cu
    detail/ops/VolumeRender.cu
    detail/ops/RayImplicitIntersection.cu
    detail/ops/SegmentsAlongRays.cu
    detail/ops/BuildDeviceGrid.cu
    detail/ops/CountEnabledVoxels.cu
    detail/ops/VoxelNeighborhood.cu
    detail/ops/SampleGridTrilinearWithGrad.cu
    detail/ops/CoordsInGrid.cu
    detail/ops/gsplat/GaussianComputeNanInfMask.cu
    detail/ops/gsplat/GaussianSplatSparse.cu
    detail/ops/gsplat/GaussianProjectionJaggedBackward.cu
    detail/ops/gsplat/GaussianProjectionForward.cu
    detail/ops/gsplat/SphericalHarmonics.cu
    detail/ops/gsplat/GaussianRasterizeBackward.cu
    detail/ops/gsplat/GaussianProjectionBackward.cu
    detail/ops/gsplat/GaussianRasterizeForward.cu
    detail/ops/gsplat/GaussianProjectionJaggedForward.cu
    detail/ops/gsplat/GaussianTileIntersection.cu
    detail/ops/UpsampleGridNearest.cu
    detail/ops/DownsampleGridAvgPool.cu
    detail/ops/JIdxForJOffsets.cu
    detail/ops/SampleGridBezierWithGrad.cu
    detail/ops/JaggedTensorIndex.cu
    detail/ops/SampleRaysUniform.cu
    detail/ops/FillFromGrid.cu
    detail/ops/SplatIntoGridTrilinear.cu
    detail/ops/DownsampleGridMaxPool.cu
    detail/ops/SampleGridBezier.cu
    detail/ops/PointsInGrid.cu
    detail/ops/IjkToIndex.cu
    detail/ops/ActiveGridGoords.cu
    detail/ops/SetMasked.cu
    detail/ops/IjkToInvIndex.cu
    detail/ops/ReadFromDense.cu
    detail/ops/VoxelsForGridBuilding.cu
    detail/ops/EnabledMask.cu
    detail/ops/ActiveVoxelsInBoundsMask.cu
    detail/ops/JIdxForGrid.cu
    detail/ops/VoxelsAlongRays.cu
    detail/ops/SampleGridTrilinear.cu
    detail/ops/SplatIntoGridBezier.cu
    detail/ops/ScaledDotProductAttention.cu
    detail/ops/PaddedIJKForMesh.cu
    detail/ops/SampleGridBezierWithGradBackward.cu
    detail/ops/convolution/pack_info/ConvolutionKernelMap.cu
    detail/ops/convolution/pack_info/BrickHaloBuffer.cu
    detail/ops/convolution/pack_info/IGEMMBitOperations.cu
    detail/ops/convolution/backend/MESparseConvolution.cu
    detail/ops/convolution/backend/SparseConvolutionCutlass.cu
    detail/ops/convolution/backend/SparseConvolutionLggs.cu
    detail/ops/convolution/backend/SparseConvolutionHalo.cu
    detail/ops/convolution/backend/SparseConvolutionImplicitGEMMGradSorted.cu
    detail/ops/convolution/backend/SparseConvolutionImplicitGEMMSorted.cu
    detail/ops/convolution/backend/SparseConvolutionImplicitGEMMGrad.cu
    detail/ops/convolution/backend/SparseConvolutionKernelMap.cu
    detail/ops/convolution/backend/SparseConvolutionImplicitGEMM.cu
    detail/ops/convolution/backend/SparseConvolutionHaloGrad.cu
    detail/ops/jagged/JaggedSort.cu
    detail/ops/jagged/JaggedReduce.cu
    detail/GridBatchImpl.cu)

### Build library
add_library(fvdb SHARED ${FVDB_CPP_FILES} ${FVDB_CU_FILES})

### Target properties
set_target_properties(
    fvdb
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "fvdb: TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

target_include_directories(fvdb PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../nanovdb
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS})
target_include_directories(fvdb PRIVATE
    ${TORCH_SOURCE_INCLUDE_DIRS}
    ${tinyply_SOURCE_DIR}/source)

# Needed by Torch for CUDA extensions
set(TORCH_CUDA_COMMON_FLAGS
    "-D__CUDA_NO_HALF_OPERATORS__"
    "-D__CUDA_NO_HALF_CONVERSIONS__"
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
    "-D__CUDA_NO_HALF2_OPERATORS__"
    "--expt-relaxed-constexpr")

target_compile_options(fvdb PRIVATE
    $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-G -Xcompiler=-O0>
    $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CXX>>:-O0>
    $<$<COMPILE_LANGUAGE:CXX>:
        "-Wno-unknown-pragmas"
        "-Wno-class-memaccess"
        "-fdiagnostics-color=always"
        "-DNANOVDB_USE_BLOSC"
        "-fvisibility=default"
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        "--extended-lambda"
        "-Xfatbin=-compress-all"
        "-fvisibility=default"
        ${TORCH_CUDA_COMMON_FLAGS}
    >)

target_link_libraries(fvdb PRIVATE
    ${TORCH_LIBRARIES} ${Python3_LIBRARIES} tinyply glm::glm cudnn_frontend cudnn cutlass blosc::blosc)

### Install rules
install(TARGETS fvdb
    EXPORT fvdb-targets
    LIBRARY DESTINATION lib)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    DESTINATION include)

### Add tests
if(FVDB_BUILD_TESTS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_google_test.cmake)
    # include CTest module -- automatically calls enable_testing()
    include(CTest)

    # Always print verbose output when tests fail if run using `make test`.
    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
    add_subdirectory(tests)
endif()

### Add benchmarks
if(FVDB_BUILD_BENCHMARKS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_google_benchmark.cmake)
    add_subdirectory(benchmarks)
endif()
