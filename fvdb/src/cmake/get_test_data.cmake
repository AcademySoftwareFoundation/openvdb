# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set(FVDB_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/fvdb_example_data")

# Download test data from github
CPMAddPackage(
  NAME fvdb_test_data
  GITHUB_REPOSITORY voxel-foundation/fvdb-test-data
  GIT_TAG main
  DOWNLOAD_ONLY YES
  SOURCE_DIR "${FVDB_TEST_DATA_DIR}"
)

if(fvdb_test_data_ADDED)
  message(STATUS "Downloaded test data to: ${fvdb_test_data_SOURCE_DIR}")
else()
  unset(FVDB_TEST_DATA_DIR)
  message(FATAL_ERROR "Failed to download test data")
endif()
