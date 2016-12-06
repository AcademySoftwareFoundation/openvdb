# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Targets to run after build
BUILD_POST_TARGETS += build-openvdb strip-rpath

# Clean build area not just removing BUILD_ROOT
CLEAN_TARGETS += clean-openvdb

# Set the package root to this path and not the one above.
PACKAGE_ROOT := $(PWD)
include ../Version.mk

.PHONY: build-openvdb
build-openvdb:
	@pybuild2 --install -DINSTALL_BASE=$(BUILD_ROOT) 

.PHONY: strip-rpath
strip-rpath:
	# stripping rpath from libs
	@cd $(BUILD_ROOT)/lib && \
	find -name "*.so*" -exec patchelf --set-rpath '' {} \;
	# Fixing executables
	@cd $(BUILD_ROOT)/bin && \
	find -name "vdb_*" -exec patchelf --set-rpath '$$ORIGIN/../lib' {} \;
	# Fixing python libs
	@cd $(BUILD_ROOT)/python/lib && \
	find -name "*.so*" -exec patchelf --set-rpath '$$ORIGIN/../../../lib' {} \;

.PHONY: clean-openvdb
clean-openvdb:
	@echo "cleaning stand alone library..."
	@pybuild2 --clean -DINSTALL_BASE=$(BUILD_ROOT) 
