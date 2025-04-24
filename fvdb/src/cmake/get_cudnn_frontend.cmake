# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME cudnn_frontend
    GITHUB_REPOSITORY NVIDIA/cudnn-frontend
    GIT_TAG v1.3.0
    OPTIONS
    "CUDNN_FRONTEND_BUILD_SAMPLES OFF"
    "CUDNN_FRONTEND_BUILD_UNIT_TESTS OFF"
    "CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF"
)
