# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Get maya and houdini versions to build for
supports_field := $(shell pk manifest --field supports)
houdini_versions := $(patsubst houdini-%,%,$(filter houdini-%,$(supports_field)))
openvdb_houdini_targets := $(addprefix build-houdini-,$(houdini_versions))
BUILD_POST_TARGETS += $(openvdb_houdini_targets) 

# Include the bulk of the makefiles
# Set the package root to this path and not the one above.
PACKAGE_ROOT := $(PWD)
PRIVATE_DIR = $(PACKAGE_ROOT)/../private
include ../Header.mk

CMAKE_VERSION=3.6.2
GCC_VERSION=4.8.3

# Note h15 needs to be built against tbb 4.3
OPENVDB_VERSION = 4.0.1_abi3

TOOLS_PACKAGE_ROOT = $(DD_TOOLS_ROOT)/$(DD_OS)/package
CMAKE_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/cmake/$(CMAKE_VERSION)
GCC_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/gcc/$(GCC_VERSION)
OPENVDB_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/openvdb/$(OPENVDB_VERSION)

ARGS += -DCMAKE_INSTALL_PREFIX=$(DIST_DIR)
ARGS += -DCMAKE_CXX_COMPILER=$(GCC_PACKAGE_ROOT)/bin/g++
ARGS += -DCMAKE_C_COMPILER=$(GCC_PACKAGE_ROOT)/bin/gcc
ARGS += -DCMAKE_CXX_FLAGS=-std=c++11
ARGS += -DCMAKE_BUILD_TYPE=Release

ifdef VERBOSE
  ARGS += -DCMAKE_VERBOSE_MAKEFILE=ON
endif

ARGS += -DOPENVDB_BUILD_CORE=OFF
ARGS += -DOPENVDB_ENABLE_RPATH=OFF
ARGS += -DOPENVDB_BUILD_HOUDINI_SOPS=ON
ARGS += -DOPENVDB_HOUDINI_SHORT_VERSION=OFF
ARGS += -DOPENVDB_HOUDINI_SUBDIR=ON
ARGS += -DOPENVDB_HOUDINI_INSTALL_LIBRARY=ON
ARGS += -DOPENVDB_LOCATION=$(OPENVDB_PACKAGE_ROOT)
ARGS += -DHDK_AUTO_GENERATE_SESITAG=OFF

# Build the houdini libraries
.PHONY:build-houdini-%
build-houdini-%: 
	@echo "building for Houdini $*"
	mkdir -p $(BUILD_DIR)/$*
	pushd $(TOOLS_PACKAGE_ROOT)/houdini/$* && \
	source ./houdini_setup  && popd && \
        cd $(BUILD_DIR)/$* &&\
		$(CMAKE_PACKAGE_ROOT)/bin/cmake $(ARGS) $(PACKAGE_ROOT)/.. && \
		make -j $(shell nproc) && make install


