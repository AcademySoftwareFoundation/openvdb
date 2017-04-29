# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Defines the version and location of the dependency packages
include ../BuildConf.mk

# Get maya and houdini versions to build for
houdini_versions = $(call _get_version,houdini,$(supports_field))
BUILD_POST_TARGETS += $(addprefix build-houdini-,$(houdini_versions))

# Include the bulk of the makefiles
# Set the package root to this path and not the one above.
PACKAGE_ROOT = $(PWD)
PRIVATE_DIR = $(PROJECT_ROOT)/private


include ../Project.mk


ARGS += $(CMAKE_DEFAULT_ARGS)

ARGS += -DOPENVDB_BUILD_CORE=OFF
ARGS += -DOPENVDB_BUILD_MAYA_PLUGIN=OFF
ARGS += -DOPENVDB_BUILD_HOUDINI_SOPS=ON
ARGS += -DOPENVDB_BUILD_PYTHON_MODULE=OFF

ARGS += -DOPENVDB_ENABLE_RPATH=OFF
ARGS += -DOPENVDB_HOUDINI_SHORT_VERSION=OFF
ARGS += -DOPENVDB_HOUDINI_SUBDIR=ON
ARGS += -DOPENVDB_HOUDINI_INSTALL_LIBRARY=ON
ARGS += -DOPENVDB_LOCATION=$(OPENVDB_PACKAGE_ROOT)
ARGS += -DHDK_AUTO_GENERATE_SESITAG=OFF
ARGS += -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--as-needed

# Build the houdini libraries
.PHONY:build-houdini-%
build-houdini-%: 
	@echo "building for Houdini $*"
	mkdir -p $(BUILD_DIR)/$*
	pushd $(TOOLS_PACKAGE_ROOT)/houdini/$* && \
	source ./houdini_setup  && popd && \
        cd $(BUILD_DIR)/$* &&\
		$(CMAKE_PACKAGE_ROOT)/bin/cmake $(ARGS) $(PROJECT_ROOT) && \
		make -j $(shell nproc) VERBOSE=1 && make install

add-houdini-version-%: build-houdini-%
	@echo -e "\nNow run: pkappend --rem=all $(DIST_DIR)/houdini/$* $(DD_OS)"\
                 "$(PACKAGE_NAME) $(PACKAGE_VERSION) houdini $*"

