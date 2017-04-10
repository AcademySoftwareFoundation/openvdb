# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Simplest possible Makefile, which provides all the services in the included
# Makefile.tdpackage. To see all the options provided, just type 'make'.
# It is recommend to add comments for new things you would be adding
# to keep this file readable for other people who may need to maintain
# your package in the future.

# The version number of the facility's Makefile this package is making use of.
USE_MAKEFILE_VERSION = 4

# Openvdb's docs are online
NO_DOCS=true

# No support for git so disabling it
NO_TAG=true

# This Makefile defines various build targets and variables, taking into
# account your settings, above. Do not remove this line unless you really
# know what you are doing.
include $(DD_TOOLS_ROOT)/etc/Makefile.tdpackage

# Path to the top of the repository
PROJECT_ROOT = $(PACKAGE_ROOT)/../..

CMAKE_DEFAULT_ARGS += -DCMAKE_INSTALL_PREFIX=$(DIST_DIR)
CMAKE_DEFAULT_ARGS += -DCMAKE_CXX_COMPILER=$(GCC_PACKAGE_ROOT)/bin/g++
CMAKE_DEFAULT_ARGS += -DCMAKE_C_COMPILER=$(GCC_PACKAGE_ROOT)/bin/gcc
CMAKE_DEFAULT_ARGS += -DCMAKE_CXX_FLAGS=-std=c++11
CMAKE_DEFAULT_ARGS += -DCMAKE_BUILD_TYPE=Release

ifdef VERBOSE
  CMAKE_DEFAULT_ARGS += -DCMAKE_VERBOSE_MAKEFILE=ON
endif
