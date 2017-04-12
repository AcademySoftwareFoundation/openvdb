# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Defines the version and location of the dependency packages
include ../BuildConf.mk

# Get maya and houdini versions to build for
maya_versions = $(call _get_version,maya,$(supports_field))
BUILD_POST_TARGETS += $(addprefix build-maya-,$(maya_versions))

# Include the bulk of the makefiles
PACKAGE_ROOT = $(PWD)
PRIVATE_DIR = $(PROJECT_ROOT)/private


include ../Project.mk


ARGS += $(CMAKE_DEFAULT_ARGS)

ARGS += -DOPENVDB_BUILD_CORE=OFF
ARGS += -DOPENVDB_BUILD_MAYA_PLUGIN=ON
ARGS += -DOPENVDB_BUILD_HOUDINI_SOPS=OFF
ARGS += -DOPENVDB_BUILD_PYTHON_MODULE=OFF

ARGS += -DOPENVDB_ENABLE_RPATH=OFF
ARGS += -DOPENVDB_MAYA_SUBDIR=ON
ARGS += -DOPENVDB_MAYA_INSTALL_MOD=OFF
ARGS += -DOPENVDB_LOCATION=$(OPENVDB_PACKAGE_ROOT)

ARGS += -DILMBASE_NAMESPACE_VERSIONING=OFF
ARGS += -DMINIMUM_BOOST_VERSION=$(BOOST_SHORT_VERSION)

ARGS += -DBOOST_ROOT=$(BOOST_PACKAGE_ROOT)
ARGS += -DILMBASE_LOCATION=$(ILMBASE_PACKAGE_ROOT)
ARGS += -DMAYA_LOCATION=$(TOOLS_PACKAGE_ROOT)/maya/$*

.PHONY: build-maya-%
build-maya-%:
	@echo "building for Maya $*"
	mkdir -p $(BUILD_DIR)/$*
	cd $(BUILD_DIR)/$* &&\
		$(CMAKE_PACKAGE_ROOT)/bin/cmake $(ARGS) $(PROJECT_ROOT) && \
		make -j $(shell nproc) && make install

