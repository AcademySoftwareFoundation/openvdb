# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
option(NANOVDB_BUILD_PYTHON_UNITTESTS [=[
  "Include the NanoVDB Python unit test. Requires a python interpreter]=]
${NANOVDB_BUILD_UNITTESTS})

option(NANOVDB_BUILD_PYTHON_STUBS
  "Generate .pyi type stubs for the nanovdb Python module (requires nanobind 2.x or newer)."
  ON)

nanobind_add_module(nanovdb_python NB_STATIC
    NanoVDBModule.cc
    PyCreateNanoGrid.cc
    PyGridChecksum.cc
    PyGridHandle.cc
    PyGridStats.cc
    PyGridValidator.cc
    PyHostBuffer.cc
    PyIO.cc
    PyMath.cc
    PyNanoToOpenVDB.cc
    PyPrimitives.cc
    PySampleFromVoxels.cc
    PyTools.cc
    cuda/PyDeviceBuffer.cc
    cuda/PyDeviceGridHandle.cu
    cuda/PyPointsToGrid.cu
    cuda/PySampleFromVoxels.cu
    cuda/PySignedFloodFill.cu
)

target_include_directories(nanovdb_python PRIVATE ${CUDA_INCLUDE_DIRECTORY})
target_link_libraries(nanovdb_python PRIVATE nanovdb ${CUDA_LIBRARIES} ${NANOVDB_BLOSC} ${NANOVDB_ZLIB} ${NANOVDB_OPENVDB} ${NANOVDB_TBB})
target_compile_definitions(nanovdb_python PRIVATE ${NANOVDB_USE_CUDA_FLAG} ${NANOVDB_USE_BLOSC_FLAG} ${NANOVDB_USE_ZLIB_FLAG} ${NANOVDB_USE_OPENVDB_FLAG} ${NANOVDB_USE_TBB_FLAG})
set_target_properties(nanovdb_python PROPERTIES OUTPUT_NAME "nanovdb")
if(SKBUILD)
  set_target_properties(nanovdb_python PROPERTIES INSTALL_RPATH "$ORIGIN/../../openvdb/lib")
  install(TARGETS nanovdb_python DESTINATION ${NANOVDB_INSTALL_LIBDIR})
  install(FILES __init__.py DESTINATION nanovdb)
  # Ship the nanovdb C/C++ headers inside the Python wheel so downstream
  # extension authors can compile against the same NanoVDB the wheel was
  # built with. nanovdb.get_include() resolves to this directory at runtime.
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
          DESTINATION nanovdb/include/nanovdb
          FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.cuh"
            PATTERN "python" EXCLUDE
            PATTERN "examples" EXCLUDE
            PATTERN "unittest" EXCLUDE
            PATTERN "cmd" EXCLUDE
            PATTERN "docs" EXCLUDE)
else()
  install(TARGETS nanovdb_python DESTINATION ${VDB_PYTHON_INSTALL_DIRECTORY})
endif()

# .pyi type stubs for IDE / type-checker support. Driven by the CMake helper
# that ships with nanobind 2.x; silently skipped on older nanobind where the
# helper does not exist.
if(NANOVDB_BUILD_PYTHON_STUBS AND COMMAND nanobind_add_stub)
  set(NANOVDB_STUB_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  nanobind_add_stub(nanovdb_python_stub
    MODULE nanovdb
    OUTPUT "${NANOVDB_STUB_OUTPUT_DIR}/nanovdb.pyi"
    PYTHON_PATH $<TARGET_FILE_DIR:nanovdb_python>
    DEPENDS nanovdb_python
    MARKER_FILE "${NANOVDB_STUB_OUTPUT_DIR}/py.typed"
    INCLUDE_PRIVATE)
  if(SKBUILD)
    install(FILES
        "${NANOVDB_STUB_OUTPUT_DIR}/nanovdb.pyi"
        "${NANOVDB_STUB_OUTPUT_DIR}/py.typed"
      DESTINATION nanovdb)
  endif()
elseif(NANOVDB_BUILD_PYTHON_STUBS)
  message(STATUS
    "NANOVDB_BUILD_PYTHON_STUBS is ON but nanobind_add_stub is unavailable "
    "(nanobind >= 2.0 required). Skipping stub generation.")
endif()

# pytest
if(NANOVDB_BUILD_PYTHON_UNITTESTS)

  set(NANOVDB_PYTHON_WORKING_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  if(WIN32)
    set(NANOVDB_PYTHON_WORKING_DIR "${NANOVDB_PYTHON_WORKING_DIR}/$<CONFIG>")
  endif()

  add_test(NAME pytest_nanovdb
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/TestNanoVDB.py -v
    WORKING_DIRECTORY "${NANOVDB_PYTHON_WORKING_DIR}")

  if(WIN32)
    set(PYTHONPATH "$ENV{PYTHONPATH};${NANOVDB_PYTHON_WORKING_DIR}")
    string(REPLACE "\\;" ";" PYTHONPATH "${PYTHONPATH}")
    string(REPLACE ";" "\\;" PYTHONPATH "${PYTHONPATH}")
    set_tests_properties(pytest_nanovdb PROPERTIES ENVIRONMENT "PYTHONPATH=${PYTHONPATH}")
  else()
    set_tests_properties(pytest_nanovdb PROPERTIES ENVIRONMENT "PYTHONPATH=$ENV{PYTHONPATH}:${NANOVDB_PYTHON_WORKING_DIR}")
  endif()
endif()
