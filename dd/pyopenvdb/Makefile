# Confidential and Proprietary Source Code
# Copyright (c) [2016] Digital Domain Productions, Inc. All rights reserved.

# Defines the version and location of the dependency packages
include ../BuildConf.mk

# Targets to run after build
python_versions = $(call _get_version,python,$(supports_field))
BUILD_POST_TARGETS += $(addprefix build-pyopenvdb-,$(python_versions))

# Set the package root to this path and not the one above.
PACKAGE_ROOT = $(PWD)
PRIVATE_DIR = $(PROJECT_ROOT)/private


include ../Project.mk

# Python need the short version for the include
PYTHON_MAJOR_VERSION = $(word 1,$(subst ., ,$*))
PYTHON_MINOR_VERSION = $(word 2,$(subst ., ,$*))
PYTHON_SHORT_VERSION = $(PYTHON_MAJOR_VERSION).$(PYTHON_MINOR_VERSION)

PYTHON_PACKAGE_ROOT = $(TOOLS_PACKAGE_ROOT)/python/$*

ARGS += $(CMAKE_DEFAULT_ARGS)
ARGS += -DPYTHON_LIBRARY=$(PYTHON_PACKAGE_ROOT)/lib/libpython$(PYTHON_SHORT_VERSION).so
ARGS += -DPYTHON_INCLUDE=$(PYTHON_PACKAGE_ROOT)/include/python$(PYTHON_SHORT_VERSION)
ARGS += -DILMBASE_NAMESPACE_VERSIONING=OFF
ARGS += -DOPENEXR_NAMESPACE_VERSIONING=OFF
ARGS += -DMINIMUM_BOOST_VERSION=$(BOOST_SHORT_VERSION)

ARGS += -DOPENVDB_ENABLE_RPATH=OFF

ARGS += -DOPENVDB_BUILD_CORE=OFF
ARGS += -DOPENVDB_BUILD_MAYA_PLUGIN=OFF
ARGS += -DOPENVDB_BUILD_HOUDINI_SOPS=OFF
ARGS += -DOPENVDB_BUILD_PYTHON_MODULE=ON

ARGS += -DPYOPENVDB_INSTALL_DIRECTORY=python/$(PYTHON_SHORT_VERSION)
ARGS += -DBOOST_ROOT=$(BOOST_PACKAGE_ROOT)
ARGS += -DZLIB_ROOT=$(ZLIB_PACKAGE_ROOT)

ARGS += -DTBB_LOCATION=$(TBB_PACKAGE_ROOT)
ARGS += -DILMBASE_LOCATION=$(ILMBASE_PACKAGE_ROOT)
ARGS += -DOPENEXR_LOCATION=$(OPENEXR_PACKAGE_ROOT)
ARGS += -DBLOSC_LOCATION=$(BLOSC_PACKAGE_ROOT)
ARGS += -DOPENVDB_LOCATION=$(OPENVDB_PACKAGE_ROOT)

.PHONY: build-pyopenvdb-%
build-pyopenvdb-%:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) &&\
		$(CMAKE_PACKAGE_ROOT)/bin/cmake $(ARGS) $(PROJECT_ROOT) && \
		make -j $(shell nproc) && make install
