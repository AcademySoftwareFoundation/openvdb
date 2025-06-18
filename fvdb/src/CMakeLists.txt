# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(fvdb LANGUAGES CXX CUDA)

# Get dependencies
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_cpm.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_nvtx.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_torch.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_tinyply.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_cutlass.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_cudnn_frontend.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_blosc.cmake)

# Build options
option(FVDB_BUILD_TESTS "Configure CMake to build tests" ON)
option(FVDB_BUILD_BENCHMARKS "Configure CMake to build (google & nvbench) benchmarks" OFF)

message(STATUS "FVDB: Configure CMake to build tests: ${FVDB_BUILD_TESTS}")
message(STATUS "FVDB: Configure CMake to build (google & nvbench) benchmarks: ${FVDB_BUILD_BENCHMARKS}")

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Symlink the latest compile_commands.json from the build dir to the source dir for clangd
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
)

# Source files
set(FVDB_CPP_FILES
    fvdb/Config.cpp
    fvdb/FVDB.cpp
    fvdb/GaussianSplatting.cpp
    fvdb/GridBatch.cpp
    fvdb/JaggedTensor.cpp
    fvdb/SparseConvPackInfo.cpp
    fvdb/detail/autograd/Attention.cpp
    fvdb/detail/autograd/AvgPoolGrid.cpp
    fvdb/detail/autograd/GaussianRender.cpp
    fvdb/detail/autograd/JaggedReduce.cpp
    fvdb/detail/autograd/MaxPoolGrid.cpp
    fvdb/detail/autograd/EvaluateSphericalHarmonics.cpp
    fvdb/detail/autograd/ReadIntoDense.cpp
    fvdb/detail/autograd/SampleGrid.cpp
    fvdb/detail/autograd/SparseConvolutionHalo.cpp
    fvdb/detail/autograd/SplatIntoGrid.cpp
    fvdb/detail/autograd/TransformPoints.cpp
    fvdb/detail/autograd/UpsampleGrid.cpp
    fvdb/detail/autograd/VolumeRender.cpp
    fvdb/detail/io/LoadNanovdb.cpp
    fvdb/detail/io/SaveNanoVDB.cpp
    fvdb/detail/TorchDeviceBuffer.cpp)

set(FVDB_CU_FILES
    fvdb/detail/ops/BuildPaddedGrid.cu
    fvdb/detail/ops/BuildGridFromNearestVoxelsToPoints.cu
    fvdb/detail/ops/BuildGridForConv.cu
    fvdb/detail/ops/BuildGridFromPoints.cu
    fvdb/detail/ops/BuildGridFromMesh.cu
    fvdb/detail/ops/BuildFineGridFromCoarse.cu
    fvdb/detail/ops/BuildDilatedGrid.cu
    fvdb/detail/ops/BuildMergedGrids.cu
    fvdb/detail/ops/BuildDenseGrid.cu
    fvdb/detail/ops/BuildCoarseGridFromFine.cu
    fvdb/detail/ops/BuildGridFromIjk.cu
    fvdb/detail/ops/CoarseIjkForFineGrid.cu
    fvdb/detail/ops/MarchingCubes.cu
    fvdb/detail/ops/ReadIntoDense.cu
    fvdb/detail/ops/JOffsetsFromJIdx.cu
    fvdb/detail/ops/TransformPointToGrid.cu
    fvdb/detail/ops/CubesInGrid.cu
    fvdb/detail/ops/JCat0.cu
    fvdb/detail/ops/GridEdgeNetwork.cu
    fvdb/detail/ops/SampleGridTrilinearWithGradBackward.cu
    fvdb/detail/ops/VolumeRender.cu
    fvdb/detail/ops/RayImplicitIntersection.cu
    fvdb/detail/ops/SegmentsAlongRays.cu
    fvdb/detail/ops/PopulateGridMetadata.cu
    fvdb/detail/ops/VoxelNeighborhood.cu
    fvdb/detail/ops/SampleGridTrilinearWithGrad.cu
    fvdb/detail/ops/CoordsInGrid.cu
    fvdb/detail/ops/gsplat/GaussianComputeNanInfMask.cu
    fvdb/detail/ops/gsplat/GaussianSplatSparse.cu
    fvdb/detail/ops/gsplat/GaussianProjectionJaggedBackward.cu
    fvdb/detail/ops/gsplat/GaussianProjectionForward.cu
    fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsForward.cu
    fvdb/detail/ops/gsplat/GaussianSphericalHarmonicsBackward.cu
    fvdb/detail/ops/gsplat/GaussianRasterizeBackward.cu
    fvdb/detail/ops/gsplat/GaussianProjectionBackward.cu
    fvdb/detail/ops/gsplat/GaussianRasterizeForward.cu
    fvdb/detail/ops/gsplat/GaussianProjectionJaggedForward.cu
    fvdb/detail/ops/gsplat/GaussianTileIntersection.cu
    fvdb/detail/ops/UpsampleGridNearest.cu
    fvdb/detail/ops/DownsampleGridAvgPool.cu
    fvdb/detail/ops/JIdxForJOffsets.cu
    fvdb/detail/ops/SampleGridBezierWithGrad.cu
    fvdb/detail/ops/JaggedTensorIndex.cu
    fvdb/detail/ops/SampleRaysUniform.cu
    fvdb/detail/ops/FillFromGrid.cu
    fvdb/detail/ops/SplatIntoGridTrilinear.cu
    fvdb/detail/ops/DownsampleGridMaxPool.cu
    fvdb/detail/ops/SampleGridBezier.cu
    fvdb/detail/ops/PointsInGrid.cu
    fvdb/detail/ops/IjkToIndex.cu
    fvdb/detail/ops/ActiveGridGoords.cu
    fvdb/detail/ops/IjkToInvIndex.cu
    fvdb/detail/ops/ReadFromDense.cu
    fvdb/detail/ops/NearestIjkForPoints.cu
    fvdb/detail/ops/ActiveVoxelsInBoundsMask.cu
    fvdb/detail/ops/JIdxForGrid.cu
    fvdb/detail/ops/VoxelsAlongRays.cu
    fvdb/detail/ops/SampleGridTrilinear.cu
    fvdb/detail/ops/SplatIntoGridBezier.cu
    fvdb/detail/ops/ScaledDotProductAttention.cu
    fvdb/detail/ops/IjkForMesh.cu
    fvdb/detail/ops/SampleGridBezierWithGradBackward.cu
    fvdb/detail/ops/convolution/pack_info/ConvolutionKernelMap.cu
    fvdb/detail/ops/convolution/pack_info/BrickHaloBuffer.cu
    fvdb/detail/ops/convolution/pack_info/IGEMMBitOperations.cu
    fvdb/detail/ops/convolution/backend/MESparseConvolution.cu
    fvdb/detail/ops/convolution/backend/SparseConvolutionCutlass.cu
    fvdb/detail/ops/convolution/backend/SparseConvolutionLggs.cu
    fvdb/detail/ops/convolution/backend/SparseConvolutionHalo.cu
    fvdb/detail/ops/convolution/backend/SparseConvolutionImplicitGEMMGradSorted.cu
    fvdb/detail/ops/convolution/backend/SparseConvolutionImplicitGEMMSorted.cu
    fvdb/detail/ops/convolution/backend/SparseConvolutionImplicitGEMMGrad.cu
    fvdb/detail/ops/convolution/backend/SparseConvolutionKernelMap.cu
    fvdb/detail/ops/convolution/backend/SparseConvolutionImplicitGEMM.cu
    fvdb/detail/ops/convolution/backend/SparseConvolutionHaloGrad.cu
    fvdb/detail/ops/jagged/JaggedSort.cu
    fvdb/detail/ops/jagged/JaggedReduce.cu
    fvdb/detail/GridBatchImpl.cu)

# Build library
add_library(fvdb SHARED ${FVDB_CPP_FILES} ${FVDB_CU_FILES})

# Target properties
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
    "-Wall"
    "-Werror"
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
    "-Wno-unknown-pragmas"
    "--extended-lambda"
    "-Xfatbin=-compress-all"
    "-fvisibility=default"
    "-Werror=all-warnings"
    "-Xcompiler=-Wall,-Werror"
    ${TORCH_CUDA_COMMON_FLAGS}
    >)

target_link_libraries(fvdb PRIVATE
    ${TORCH_LIBRARIES} ${Python3_LIBRARIES} tinyply cudnn_frontend cudnn cutlass blosc::blosc)

# Install rules
install(TARGETS fvdb
    EXPORT fvdb-targets
    LIBRARY DESTINATION lib)

# Use pattern exclusion to prevent recursion
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fvdb/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
    PATTERN "*/*" EXCLUDE)

# Add tests subdirectory if tests are enabled
if(FVDB_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Add benchmarks
if(FVDB_BUILD_BENCHMARKS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_google_benchmark.cmake)
    add_subdirectory(benchmarks)
endif()
