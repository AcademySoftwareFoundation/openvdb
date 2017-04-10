# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Targets to run after build
BUILD_POST_TARGETS += build-openvdb 
#strip-rpath


CMAKE_VERSION=3.6.2
GCC_VERSION=4.8.3
BOOST_VERSION = 1.61.0
TBB_VERSION = 4.4.6
OPENEXR_VERSION = 2.2.0
ILMBASE_VERSION = 2.2.0
CPPUNIT_VERSION = 1.12.1
GLFW_VERSION = 2.7.6
BLOSC_VERSION = 1.5.0
PYTHON_VERSION = 2.7.3
PYTHON_MAJOR_VERSION = $(word 1,$(subst ., ,$(PYTHON_VERSION)))
PYTHON_MINOR_VERSION = $(word 2,$(subst ., ,$(PYTHON_VERSION)))
PYTHON_MM_VERSION=$(PYTHON_MAJOR_VERSION).$(PYTHON_MINOR_VERSION)

TOOLS_PACKAGE_ROOT = $(DD_TOOLS_ROOT)/$(DD_OS)/package
CMAKE_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/cmake/$(CMAKE_VERSION)
GCC_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/gcc/$(GCC_VERSION)
BOOST_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/boost/$(BOOST_VERSION)
TBB_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/tbb/$(TBB_VERSION)
ILMBASE_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/ilmbase/$(ILMBASE_VERSION)
OPENEXR_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/openexr/$(OPENEXR_VERSION)
CPPUNIT_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/cppunit/$(CPPUNIT_VERSION)
GLFW_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/glfw/$(GLFW_VERSION)
PYTHON_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/python/$(PYTHON_VERSION)
BLOSC_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/blosc/$(BLOSC_VERSION)
# Set the package root to this path and not the one above.
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

ARGS += -DPYTHON_LIBRARY=$(PYTHON_PACKAGE_ROOT)/lib/libpython$(PYTHON_MM_VERSION).so
ARGS += -DPYTHON_INCLUDE=$(PYTHON_PACKAGE_ROOT)/include/python$(PYTHON_MM_VERSION)
ARGS += -DILMBASE_NAMESPACE_VERSIONING=OFF
ARGS += -DOPENEXR_NAMESPACE_VERSIONING=OFF
ARGS += -DMINIMUM_BOOST_VERSION=1.55

ARGS += -DOPENVDB_ENABLE_RPATH=OFF

ifeq ($(DISABLE_3_ABI_COMPATIBLE),YES)
  ARGS += -DOPENVDB_3_ABI_COMPATIBLE=OFF
endif

ARGS += -DBOOST_ROOT=$(BOOST_PACKAGE_ROOT)
ARGS += -DTBB_LOCATION=$(TBB_PACKAGE_ROOT)
ARGS += -DILMBASE_LOCATION=$(ILMBASE_PACKAGE_ROOT)
ARGS += -DOPENEXR_LOCATION=$(OPENEXR_PACKAGE_ROOT)
ARGS += -DCPPUNIT_LOCATION=$(CPPUNIT_PACKAGE_ROOT)
ARGS += -DGLFW_LOCATION=$(GLFW_PACKAGE_ROOT)
ARGS += -DBLOSC_LOCATION=$(BLOSC_PACKAGE_ROOT)

.PHONY: build-openvdb
build-openvdb:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) &&\
		$(CMAKE_PACKAGE_ROOT)/bin/cmake $(ARGS) $(PACKAGE_ROOT)/.. && \
		make -j $(shell nproc) && make install

# .PHONY: build-openvdb
# build-openvdb:
# 	pybuild2 -vv --install -DINSTALL_BASE=$(DIST_DIR) -Dabi=3

# .PHONY: strip-rpath
# strip-rpath:
# 	# stripping rpath from libs
# 	@cd $(DIST_DIR)/lib && \
# 	find -name "*.so*" -exec patchelf --set-rpath '' {} \;
# 	# Fixing executables
# 	@cd $(DIST_DIR)/bin && \
# 	find -name "vdb_*" -exec patchelf --set-rpath '$$ORIGIN/../lib' {} \;
# 	# Fixing python libs
# 	@cd $(DIST_DIR)/python/lib && \
# 	find -name "*.so*" -exec patchelf --set-rpath '$$ORIGIN/../../../lib' {} \;

# .PHONY: clean-openvdb
# clean-openvdb:
# 	@echo "cleaning stand alone library..."
# 	@pybuild2 --clean -DINSTALL_BASE=$(DIST_DIR) 
