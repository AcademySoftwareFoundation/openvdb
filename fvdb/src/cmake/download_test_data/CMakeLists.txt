# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# This file is used to download test data for the OpenVDB project.
# It is called by the testing script to ensure that the test data is downloaded in
# the same way as when building the tests from source.

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(DownloadTestData NONE)

# Include the necessary setup scripts using relative paths from this file's location
# Adjust these paths based on the actual location relative to your project root
include(${CMAKE_CURRENT_SOURCE_DIR}/../get_cpm.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../get_test_data.cmake)
