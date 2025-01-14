# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.7.1
  OPTIONS "BENCHMARK_ENABLE_TESTING Off"
  GIT_SHALLOW TRUE
)
