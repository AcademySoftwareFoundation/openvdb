# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Get maya and houdini versions to build for
supports_field := $(shell pk manifest --field supports)
maya_versions := $(patsubst maya-%,%,$(filter maya-%,$(supports_field)))

openvdb_maya_targets := $(addprefix build-maya-,$(maya_versions))
BUILD_POST_TARGETS += $(openvdb_maya_targets)

CMAKE_VERSION=3.6.2
GCC_VERSION=4.8.3
BOOST_VERSION = 1.55.0
OPENEXR_VERSION = 2.2.0
ILMBASE_VERSION = 2.2.0
OPENVDB_VERSION = 4.0.1_abi3

TOOLS_PACKAGE_ROOT = $(DD_TOOLS_ROOT)/$(DD_OS)/package
CMAKE_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/cmake/$(CMAKE_VERSION)
GCC_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/gcc/$(GCC_VERSION)
BOOST_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/boost/$(BOOST_VERSION)
ILMBASE_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/ilmbase/$(ILMBASE_VERSION)
OPENEXR_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/openexr/$(OPENEXR_VERSION)
OPENVDB_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/openvdb/$(OPENVDB_VERSION)

################################################################################
# Include the bulk of the makefiles
PACKAGE_ROOT := $(PWD)
PRIVATE_DIR = $(PACKAGE_ROOT)/../private
include ../Header.mk

ARGS += -DCMAKE_INSTALL_PREFIX=$(DIST_DIR)
ARGS += -DCMAKE_CXX_COMPILER=$(GCC_PACKAGE_ROOT)/bin/g++
ARGS += -DCMAKE_C_COMPILER=$(GCC_PACKAGE_ROOT)/bin/gcc
ARGS += -DCMAKE_CXX_FLAGS=-std=c++11
ARGS += -DCMAKE_BUILD_TYPE=Release
ARGS += -DCMAKE_VERBOSE_MAKEFILE=ON
ARGS += -DCMAKE_VERBOSE=ON

ARGS += -DOPENVDB_BUILD_CORE=OFF
ARGS += -DOPENVDB_BUILD_MAYA_PLUGIN=ON
ARGS += -DOPENVDB_ENABLE_RPATH=OFF
ARGS += -DOPENVDB_MAYA_HYPHEN_PLUGINS=OFF
ARGS += -DOPENVDB_MAYA_SUBDIR=ON
ARGS += -DOPENVDB_MAYA_INSTALL_MOD=OFF
ARGS += -DOPENVDB_LOCATION=$(OPENVDB_PACKAGE_ROOT)

ARGS += -DILMBASE_NAMESPACE_VERSIONING=OFF
ARGS += -DMINIMUM_BOOST_VERSION=1.55

ARGS += -DBOOST_ROOT=$(BOOST_PACKAGE_ROOT)
ARGS += -DILMBASE_LOCATION=$(ILMBASE_PACKAGE_ROOT)
ARGS += -DMAYA_LOCATION=$(TOOLS_PACKAGE_ROOT)/maya/$*

.PHONY: build-maya-%
build-maya-%:
	@echo "building for Maya $*"
	mkdir -p $(BUILD_DIR)/$*
	cd $(BUILD_DIR)/$* &&\
		$(CMAKE_PACKAGE_ROOT)/bin/cmake $(ARGS) $(PACKAGE_ROOT)/.. && \
		make -j $(shell nproc) && make install

