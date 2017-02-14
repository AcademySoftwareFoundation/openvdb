# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# List of user targets to invoke after a successful build.
CLEAN_TARGETS += clean-maya

# Get maya and houdini versions to build for
supports_field := $(shell pk manifest --field supports)
maya_versions := $(patsubst maya-%,%,$(filter maya-%,$(supports_field)))

openvdb_maya_targets := $(addprefix build-maya-,$(maya_versions))
# defined in Strip.mk
strip_rpath_targets := $(addprefix strip-rpath-,$(maya_versions))

BUILD_POST_TARGETS += $(openvdb_maya_targets) $(strip_rpath_targets)

################################################################################
# Include the bulk of the makefiles
PACKAGE_ROOT := $(PWD)
include ../Header.mk

################################################################################
# Strip rpath from libs
STRIP_PATH = $(BUILD_ROOT)/maya
include ../Strip.mk

################################################################################
# Build the maya libraries

# Defines OPENVDB_PATH
include ../Variables.mk

.PHONY: build-maya-%
build-maya-%:
	@echo "building for Maya $*"
	@pybuild2 --clean -DMAYA_VERSION=$* $(OPENVDB_PATH)&& \
	pybuild2 --install  -DMAYA_VERSION=$* $(OPENVDB_PATH) \
                 -DINSTALL_BASE=$(BUILD_ROOT)

.PHONY: clean-maya
clean-maya: 
	@echo "cleaning build environment..."
	@pybuild2 --clean -DMAYA_VERSION=$* $(OPENVDB_PATH)

