# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# List of user targets to invoke after a successful build.
CLEAN_TARGETS += clean-openvdb-houdini

HOUDINI_DSO_DIR = $(BUILD_ROOT)/houdini

# Get maya and houdini versions to build for
supports_field := $(shell pk manifest --field supports)
houdini_versions := $(patsubst houdini-%,%,$(filter houdini-%,$(supports_field)))

openvdb_houdini_targets := $(addprefix build-houdini-,$(houdini_versions))
# Defined in  Strip.mk
strip_rpath_targets := $(addprefix strip-rpath-,$(houdini_versions))

BUILD_POST_TARGETS += $(openvdb_houdini_targets) $(strip_rpath_targets)

################################################################################
# Create symlinks for the major versions
HOUDINI_TARGET_VERSIONS = $(houdini_versions)
HOUDINI_DSO_DIR = $(BUILD_ROOT)/houdini
include ../../Make/HoudiniShortVersion.mk
BUILD_POST_TARGETS += $(houdini_link_targets)

################################################################################
# Include the bulk of the makefiles
PACKAGE_ROOT := $(PWD)
include ../Version.mk

################################################################################
# Strip rpaths
STRIP_PATH = $(BUILD_ROOT)/houdini
include ../Strip.mk

################################################################################
# Build the houdini libraries

# Defines OPENVDB_PATH
include ../Variables.mk

.PHONY:build-houdini-%
build-houdini-%: 
	@echo "building for Houdini $*"
	pushd $(DD_TOOLS_ROOT)/$(OS)/package/houdini/$* && \
	source ./houdini_setup  && popd && \
	pybuild2 --clean -DHOUDINI_VERSION=$* $(OPENVDB_PATH) && \
	pybuild2 --install -DHOUDINI_VERSION=$* $(OPENVDB_PATH) \
	         -DDESTDIR=$(BUILD_ROOT)/houdini/$*

add-houdini-version-%: build-houdini-%
	@echo -e "\nNow run: pkappend --rem=all $(BUILD_ROOT)/houdini/$* $(OS)"\
                 "$(PACKAGE_NAME) $(PACKAGE_VERSION) houdini $*"

################################################################################
.PHONY:clean-openvdb-houdini
clean-openvdb-houdini: 
	@echo "cleaning..."
	pushd $(DD_TOOLS_ROOT)/$(OS)/package/houdini/$(firstword $(houdini_versions)) \
	&& source ./houdini_setup  && popd && \
	pybuild2 --clean  $(OPENVDB_PATH) 
