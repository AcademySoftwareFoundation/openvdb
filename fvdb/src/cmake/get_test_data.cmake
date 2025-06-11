# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Download test data from github
CPMAddPackage(
  NAME fvdb_test_data
  GITHUB_REPOSITORY voxel-foundation/fvdb-test-data
  GIT_TAG 92f48e9cea9d47bb166f7b6831c8d88542ec1ec2
  DOWNLOAD_ONLY YES
)

if(fvdb_test_data_ADDED)
  set(FVDB_TEST_DATA_DIR "${CPM_PACKAGE_fvdb_test_data_SOURCE_DIR}")
  message(STATUS "Downloaded test data to: ${FVDB_TEST_DATA_DIR}")
else()
  unset(FVDB_TEST_DATA_DIR)
  message(FATAL_ERROR "Failed to download test data")
endif()
