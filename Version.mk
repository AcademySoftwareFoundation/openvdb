# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Simplest possible Makefile, which provides all the services in the included
# Makefile.tdpackage. To see all the options provided, just type 'make'.
# It is recommend to add comments for new things you would be adding
# to keep this file readable for other people who may need to maintain
# your package in the future.

# The version number of the facility's Makefile this package is making use of.
USE_MAKEFILE_VERSION = 3

# By default, all the directories under src, except "wrappers" and "python",
# whose contents are handled specially, will be copied to your package build
# directory private/build.  You can override this by giving a list of specific
# directories from src to copy to the build directory...
# SRC_DIRS += src/bin
# SRC_DIRS += src/maya
SRC_DIRS := 
# By default, src and all of its subdirectories will be searched for PyQT .ui
# files from which to create Python code.  You can give specific directories
# in which to look for .ui files here...
# UI_DIRS += src/foo/UI

# By default, src and all of its subdirectories will be searched for Makefiles
# to run.  You can give specific directories in which to run Makefiles here...
# COMPILE_DIRS += src/maya/plugins
# COMPILE_DIRS += src/foo
COMPILE_DIRS :=
# By default, all the files in src/wrappers will be installed as tool wrappers.
# You can override this by listing specific wrapper files here...
# WRAPPERS += src/wrappers/myTool1
# WRAPPERS += src/wrappers/myTool2

BUILD_ROOT ?= $(BUILD_DIR)

BUILD_POST_TARGETS += set-version

# If your package does not have any operating-system-dependent components (such
# as compiled plugins), you may set this variable to a space-separated list of
# operating systems you want to install to by default.  If you don't set this
# variable, the install will be for the operating system of the machine you are
# on when you call 'make install'.
# PLATFORMS = cent5_64 cent6_64 xp_32 xp_64 win7_64

NO_TAG := yes
# This Makefile defines various build targets and variables, taking into
# account your settings, above. Do not remove this line unless you really
# know what you are doing.
include $(DD_TOOLS_ROOT)/etc/Makefile.tdpackage

# Version for openvdb, overrides the dummy version in the manifests.
override PACKAGE_VERSION := 3.2.0

.PHONY: set-version
set-version:
	@echo "updating the version in the manifest..."
	@sed -Ei 's/(^Version:).*/\1 $(PACKAGE_VERSION)/' $(BUILD_ROOT)/manifest && \
	sed -Ei 's/(- openvdb-).*/\1$(PACKAGE_VERSION)/' $(BUILD_ROOT)/manifest && \
	sed -i '/^# NOTE: version/d' $(BUILD_ROOT)/manifest

