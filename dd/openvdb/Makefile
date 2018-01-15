# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Defines the version and location of the dependency packages
include ../BuildConf.mk

# Targets to run after build
BUILD_POST_TARGETS += build-openvdb 

# Set the package root to this path and not the one above.
PACKAGE_ROOT = $(PWD)
PRIVATE_DIR = $(PROJECT_ROOT)/private


include ../Project.mk


ARGS += $(CMAKE_DEFAULT_ARGS)
ARGS += -DILMBASE_NAMESPACE_VERSIONING=OFF
ARGS += -DOPENEXR_NAMESPACE_VERSIONING=OFF
ARGS += -DMINIMUM_BOOST_VERSION=$(BOOST_SHORT_VERSION)

ARGS += -DOPENVDB_ENABLE_RPATH=OFF

ARGS += -DOPENVDB_BUILD_CORE=ON
ARGS += -DOPENVDB_BUILD_MAYA_PLUGIN=OFF
ARGS += -DOPENVDB_BUILD_PYTHON_MODULE=OFF
ARGS += -DOPENVDB_BUILD_HOUDINI_SOPS=OFF
ARGS += -DUSE_GLFW3=ON
ARGS += -DGLFW3_USE_STATIC_LIBS=ON

# Disable abi 3 compatibility when the flavor tag _abi3 is not used.
ifeq ($(filter abi3,$(subst _, ,$(PACKAGE_VERSION))),)
  ARGS += -DOPENVDB_ENABLE_3_ABI_COMPATIBLE=OFF
endif

ARGS += -DBOOST_ROOT=$(BOOST_PACKAGE_ROOT)

ifneq ($(BOOST_USE_SHARED_LIBS),YES)
  ARGS += -DBoost_USE_STATIC_LIBS=ON
endif

ARGS += -DZLIB_ROOT=$(ZLIB_PACKAGE_ROOT)

ARGS += -DTBB_LOCATION=$(TBB_PACKAGE_ROOT)
ARGS += -DILMBASE_LOCATION=$(ILMBASE_PACKAGE_ROOT)
ARGS += -DOPENEXR_LOCATION=$(OPENEXR_PACKAGE_ROOT)
ARGS += -DCPPUNIT_LOCATION=$(CPPUNIT_PACKAGE_ROOT)
ARGS += -DGLFW3_LOCATION=$(GLFW_PACKAGE_ROOT)
ARGS += -DBLOSC_LOCATION=$(BLOSC_PACKAGE_ROOT)

.PHONY: build-openvdb
build-openvdb:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) &&\
		$(CMAKE_PACKAGE_ROOT)/bin/cmake $(ARGS) $(PROJECT_ROOT) && \
		make -j $(shell nproc) && make install
