# Copyright (c) 2015 Double Negative Visual Effects
#
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
#
# Redistributions of source code must retain the above copyright
# and license notice and the following restrictions and disclaimer.
#
# *     Neither the name of Double Negative Visual Effects nor the names
# of its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
# LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
#
# Makefile for OpenVDB Points Clarisse DSOs
#

# Targets:
#   clarisselib         utility library for Clarisse OpenVDB Points Modules
#   clarissemodules     Clarisse OpenVDB Points Modules (installed into CLARISSE_DESTDIR)
#
#   all                 [default target] all of the above
#   install             install clarisselib into subdirectories of DESTDIR,
#                       and the Modules into CLARISSE_DESTDIR
#   clean               delete generated files from the local directory
#   depend              recompute source file header dependencies
#
# Options:
#   abi=2               build for compatibility with the OpenVDB 2.x Grid ABI
#                       (some OpenVDB 3.x features will be disabled)
#   shared=no           link executables against static OpenVDB libraries
#                       (default: link against shared libraries)
#   debug=yes           build with debugging symbols and without optimization
#   verbose=yes         run commands (in particular, hcustom and doxygen)
#                       in verbose mode


#
# The following variables must be defined, either here or on the command line
# (e.g., "make install DESTDIR=/usr/local"):
#
# Note that the OpenVDB library and these Clarisse tools must be built
# against compatible versions of the Boost, OpenEXR and TBB libraries.
# Fortunately, all three are included in the Clarisse HDK, so the relevant
# variables below point by default to the HDK library and header directories:
# $(HDSO) and $(HT)/include, respectively.  (Source the CLARISSE_setup script
# to set those two environment variables.)
#


# The directory into which to install libraries and header files
# (it's recommended that this match the DESTDIR in the OpenVDB
# library makefile)
DESTDIR := /tmp/OpenVDB

# The root directory for Clarisse and Clarisse SDK
# (default as suggested by Clarisse's INSTALL)
CLARISSE_ROOT := ~/clarisse
CLARISSE_SDK_ROOT := ~/clarisse_sdk

# The directory into which to install Clarisse DSOs
CLARISSE_DESTDIR := $(CLARISSE_ROOT)/module

# The parent directory of the clarisse_sdk/ header directory
CLARISSE_INCL_DIR := $(CLARISSE_SDK_ROOT)/include
# The directory containing clarisse libs
CLARISSE_LIB_DIR := $(CLARISSE_ROOT)
CLARISSE_LIB := -lix_core \
                -lix_gmath \
                -lix_geometry \
                -lix_of \
                -lix_module \
                -lix_particle \
                -lix_point_cloud

# The parent directory of the openvdb/ header directory
OPENVDB_INCL_DIR := $(DESTDIR)/include
# The directory containing libopenvdb
OPENVDB_LIB_DIR := $(DESTDIR)/lib
OPENVDB_LIB := -lopenvdb

# The parent directory of the openvdb_points/ header directory
OPENVDB_POINTS_INCL_DIR := $(DESTDIR)/include
# The directory containing libopenvdb_points
OPENVDB_POINTS_LIB_DIR := $(DESTDIR)/lib
OPENVDB_POINTS_LIB :=  -lopenvdb_points

# The parent directory of the boost/ header directory
BOOST_INCL_DIR := $(CLARISSE_SDK_ROOT)/include
# The directory containing libboost_iostreams, libboost_system, etc.
BOOST_LIB_DIR := $(CLARISSE_SDK_ROOT)/lib
BOOST_LIB := -lboost_iostreams -lboost_system

# The parent directory of the OpenEXR/ header directory (which contains half.h)
ILMBASE_INCL_DIR := $(CLARISSE_SDK_ROOT)/include
# The directory containing libHalf
ILMBASE_LIB_DIR := $(CLARISSE_SDK_ROOT)/lib
ILMBASE_LIB := -lIlmThread -lIex -lImath
HALF_LIB := -lHalf

# The parent directory of the tbb/ header directory
TBB_INCL_DIR := $(CLARISSE_SDK_ROOT)/include
# The directory containing libtbb
TBB_LIB_DIR := $(CLARISSE_SDK_ROOT)/lib
TBB_LIB := -ltbb


#
# Ideally, users shouldn't need to change anything below this line.
#

SHELL = /bin/bash

# Turn off implicit rules for speed
.SUFFIXES:

# Determine the platform.
ifeq ("$(OS)","Windows_NT")
    WINDOWS_NT := 1
else
    UNAME_S := $(shell uname -s)
    ifeq ("$(UNAME_S)","Linux")
        LINUX := 1
    else
        ifeq ("$(UNAME_S)","Darwin")
            MBSD := 1
        endif
    endif
endif

CLARISSE_CMAGEN := LD_LIBRARY_PATH=$(CLARISSE_ROOT) $(CLARISSE_ROOT)/cmagen

CLARISSE_INCL_DIR := $(CLARISSE_SDK_ROOT)/include
CLARISSE_FLAGS :=

ifeq (yes,$(strip $(debug)))
    OPTIMIZE := -g
    CLARISSE_FLAGS += -g
else
    OPTIMIZE := -O3 -DNDEBUG -fexpensive-optimizations
endif

ifeq (yes,$(strip $(verbose)))
    QUIET :=
else
    QUIET := > /dev/null
endif

INCLDIRS := -I . -I .. -I $(OPENVDB_INCL_DIR) -I $(OPENVDB_POINTS_INCL_DIR) -isystem $(BOOST_INCL_DIR) -isystem $(ILMBASE_INCL_DIR) -isystem $(TBB_INCL_DIR) -I $(CLARISSE_INCL_DIR)
CLARISSE_INCLDIRS := $(subst isystem,I,$(INCLDIRS))


CXXFLAGS += -pthread $(OPTIMIZE) $(INCLDIRS) -march=nocona
ifeq (2,$(strip $(abi)))
    CXXFLAGS += -DOPENVDB_2_ABI_COMPATIBLE
endif
ifdef MBSD
    # Darwin ld treats undefined symbols as errors by default;
    # change to runtime resolution, like Linux.
    CXXFLAGS += -undefined dynamic_lookup
endif

ifneq (,$(strip $(wildcard $(CLARISSE_INCL_DIR))))
    CLARISSE_CXXFLAGS := -Wall
    CLARISSE_LDFLAGS :=
endif

LIBS := \
    -ldl -lm -lz \
    -L$(ILMBASE_LIB_DIR) $(HALF_LIB) \
    -L$(TBB_LIB_DIR) $(TBB_LIB) \
    -L$(BOOST_LIB_DIR) $(BOOST_LIB) \
    -L$(OPENVDB_LIB_DIR) $(OPENVDB_LIB) \
    -L$(OPENVDB_POINTS_LIB_DIR) $(OPENVDB_POINTS_LIB) \
    -L$(CLARISSE_LIB_DIR) $(CLARISSE_LIB) \
#
LIBS_RPATH := \
    -ldl -lm -lz \
    -Wl,-rpath,$(ILMBASE_LIB_DIR) -L$(ILMBASE_LIB_DIR) $(HALF_LIB) \
    -Wl,-rpath,$(TBB_LIB_DIR) -L$(TBB_LIB_DIR) $(TBB_LIB) \
    -Wl,-rpath,$(BOOST_LIB_DIR) -L$(BOOST_LIB_DIR) $(BOOST_LIB) \
#
ifdef LINUX
    LIBS += -lrt
    LIBS_RPATH += -lrt
endif

LIBCLARISSE_INCLUDE_NAMES := \
	clarisse/module/Geometry_OpenVDBPoints.h \
	clarisse/module/GeometryProperty_OpenVDBPoints.h \
	clarisse/module/ResourceData_OpenVDBPoints.h \
#
LIBCLARISSE_SRC_NAMES := \
	clarisse/module/Geometry_OpenVDBPoints.cc \
	clarisse/module/GeometryProperty_OpenVDBPoints.cc \
	clarisse/module/ResourceData_OpenVDBPoints.cc \
#
LIBCLARISSE_CID_NAMES := \
	clarisse/module/Geometry_OpenVDBPoints.cid \
#
CLARISSE_MODULE_SRC_NAMES := \
	clarisse/module/Module_OpenVDB_Points.cc \
#
LIBCLARISSE_CMA_NAMES := $(LIBCLARISSE_CID_NAMES:.cid=.cma)
CLARISSE_MODULE_NAMES := $(basename $(notdir $(CLARISSE_MODULE_SRC_NAMES)))

ALL_INCLUDE_FILES := \
    $(LIBCLARISSE_INCLUDE_NAMES) \
#
CLARISSE_SRC_FILES := \
    $(LIBCLARISSE_SRC_NAMES) \
    $(LIBCLARISSE_CMA_NAMES) \
    $(CLARISSE_MODULE_SRC_NAMES) \
#
ALL_SRC_FILES := $(CLARISSE_SRC_FILES)

LIBCLARISSE_OBJ_NAMES := $(LIBCLARISSE_SRC_NAMES:.cc=.o)
CLARISSE_MODULE_OBJ_NAMES := $(CLARISSE_MODULE_SRC_NAMES:.cc=.o)
CLARISSE_MODULE_SO_NAMES := $(addprefix $(CLARISSE_DESTDIR)/,\
    $(notdir $(CLARISSE_MODULE_SRC_NAMES:.cc=.so)))

LIB_MAJOR_VERSION=$(shell grep 'define OPENVDB_POINTS_LIBRARY_MAJOR_VERSION_NUMBER ' \
    $(OPENVDB_POINTS_INCL_DIR)/openvdb_points/version.h | sed 's/[^0-9]*//g')
LIB_MINOR_VERSION=$(shell grep 'define OPENVDB_POINTS_LIBRARY_MINOR_VERSION_NUMBER ' \
    $(OPENVDB_POINTS_INCL_DIR)/openvdb_points/version.h | sed 's/[^0-9]*//g')
LIB_PATCH_VERSION=$(shell grep 'define OPENVDB_POINTS_LIBRARY_PATCH_VERSION_NUMBER ' \
    $(OPENVDB_POINTS_INCL_DIR)/openvdb_points/version.h | sed 's/[^0-9]*//g')

LIB_VERSION=$(LIB_MAJOR_VERSION).$(LIB_MINOR_VERSION).$(LIB_PATCH_VERSION)

LIBCLARISSE_NAME=libopenvdb_points_clarisse
LIBCLARISSE_STATIC := $(LIBCLARISSE_NAME).a
LIBCLARISSE_SHARED := $(LIBCLARISSE_NAME).so.$(LIB_VERSION)

ifeq (no,$(strip $(shared)))
    LIBCLARISSE := $(LIBCLARISSE_STATIC)
else
    LIBCLARISSE := $(LIBCLARISSE_SHARED)

    # For debugging, pass "rpath=yes" (the default setting) to encode
    # the path to libopenvdb.so into Modules.  With "rpath=no", the directory
    # containing libopenvdb.so will need to be added to LD_LIBRARY_PATH.
    ifeq (no,$(strip $(rpath)))
        LIBOPENVDB_RPATH :=
    else
        ifdef MBSD
            LIBOPENVDB_RPATH := -install_name @rpath$(OPENVDB_LIB_DIR) -install_name @rpath$(OPENVDB_POINTS_LIB_DIR)
        else
            LIBOPENVDB_RPATH := -Wl,-rpath,$(OPENVDB_LIB_DIR) -Wl,-rpath,$(OPENVDB_POINTS_LIB_DIR)
        endif
    endif # rpath
    CLARISSE_LDFLAGS += $(LIBOPENVDB_RPATH)
endif # shared

ifeq (,$(strip $(CLARISSE_CC)))
	# If CLARISSE_CC is not set, set it to CXX if CXX is set, otherwise set it to
	# something platform-appropriate.  (This replicates the logic in hcustom.)
	ifneq (,$(strip $(CXX)))
		CLARISSE_CC = $(CXX)
	else
	    ifdef LINUX
	        CLARISSE_CC = g++
	    else ifdef MBSD
	        CLARISSE_CC = clang++
	    #else ifdef WINDOWS_NT  # TODO
	    #    CLARISSE_CC =
	    else
	        CLARISSE_CC =
	    endif
	endif
	ifeq (,$(strip $(CLARISSE_CC)))
	$(error Cannot determine which C++ compiler to use; please specify a compiler by setting $$(CXX))
	endif
endif

DEPEND := CLARISSE_deps

# Get the list of dependencies that are newer than the current target,
# but limit the list to at most three entries.
list_deps = $(if $(wordlist 4,5,$(?F)),$(firstword $(?F)) and others,$(wordlist 1,3,$(?F)))

ALL_TARGETS := \
    $(LIBCLARISSE) \
    $(DEPEND) \
    $(LIBCLARISSE_NAME).so \
#

.SUFFIXES: .o .cc

.PHONY: all clean depend clarissemodules clarisselib install

.cc.o:
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) -c $(CXXFLAGS) -fPIC -o $@ $<

all: depend clarisselib clarissemodules

# Create an openvdb_clarisse/ symlink to the clarisse/ subdirectory,
# to mirror the DWA directory structure.
openvdb_clarisse:
	ln -f -s clarisse openvdb_clarisse

$(LIBCLARISSE_CMA_NAMES): %.cma: %.cid
	@echo "Building $@ because of $(call list_deps)"
	$(CLARISSE_CMAGEN) $< -module_path "$(CLARISSE_ROOT)/module" -config_file ""

$(LIBCLARISSE_OBJ_NAMES): %.o: %.cc $(LIBCLARISSE_CMA_NAMES)
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) -c $(CXXFLAGS) -fPIC $(CLARISSE_CXXFLAGS) \
	    -isystem $(CLARISSE_INCL_DIR) -fPIC -o $@ $<

$(LIBCLARISSE_STATIC): $(LIBCLARISSE_OBJ_NAMES)
	@echo "Building $@ because of $(call list_deps)"
	$(AR) cr $@ $^

$(LIBCLARISSE_SHARED): $(LIBCLARISSE_OBJ_NAMES)
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) $(CXXFLAGS) -fPIC $(CLARISSE_LDFLAGS) -shared -o $@ $^ $(LIBS_RPATH)

$(LIBCLARISSE_NAME).so: $(LIBCLARISSE_SHARED)
	@# hcustom expects a library named "libname.so", not "libname.so.1.0".
	ln -f -s $< $@

clarisselib: $(LIBCLARISSE)
ifneq (no,$(strip $(shared)))
clarisselib: $(LIBCLARISSE_NAME).so
endif

# hcustom (see below) outputs both a .o and a .so file.
# The .o isn't dependent on $(LIBCLARISSE), but the .so is.
$(CLARISSE_MODULE_OBJ_NAMES): $(LIBCLARISSE)

$(CLARISSE_MODULE_OBJ_NAMES): %.o: %.cc
	@echo "Building $(CLARISSE_DESTDIR)/$(@:.o=.so) because of $(call list_deps)"
	mkdir -p $(CLARISSE_DESTDIR)
	$(CLARISSE_CC) -shared -o $(CLARISSE_DESTDIR)/$(@:.o=.so) $(CXXFLAGS) $(CLARISSE_FLAGS) -fPIC $(CLARISSE_LDFLAGS) $(CLARISSE_INCLDIRS) $(LIBS) \
	        -L. -lopenvdb_points_clarisse -L$(OPENVDB_LIB_DIR) $(OPENVDB_LIB) -L$(OPENVDB_POINTS_LIB_DIR) $(OPENVDB_POINTS_LIB) $< $(QUIET)

$(CLARISSE_MODULE_SO_NAMES): $(CLARISSE_DESTDIR)/%.so: clarisse/module/%.o

clarissemodules: $(CLARISSE_MODULE_SO_NAMES)

install: clarisselib clarissemodules
	mkdir -p $(DESTDIR)/include/openvdb_clarisse
	@echo "Created $(DESTDIR)/include/openvdb_clarisse"
	cp -f $(LIBCLARISSE_INCLUDE_NAMES) $(DESTDIR)/include/openvdb_clarisse/
	@#
	mkdir -p $(DESTDIR)/lib
	@echo "Created $(DESTDIR)/lib/"
	cp -f $(LIBCLARISSE) $(DESTDIR)/lib
	pushd $(DESTDIR)/lib > /dev/null; \
	    if [ -f $(LIBCLARISSE_SHARED) ]; then \
	        ln -f -s $(LIBCLARISSE_SHARED) $(LIBCLARISSE_NAME).so; fi; \
	    popd > /dev/null
	@echo "Copied libopenvdb_points_clarisse to $(DESTDIR)/lib/"

# TODO: This accumulates all source file dependencies into a single file
# containing a rule for each *.o file.  Consider generating a separate
# dependency file for each *.o file instead.
$(DEPEND): $(ALL_INCLUDE_FILES) $(ALL_SRC_FILES)
	@echo "Generating dependencies because of $(call list_deps)"
	$(RM) $(DEPEND)
	for f in $(CLARISSE_SRC_FILES); \
		do $(CXX) $(CXXFLAGS) $(CLARISSE_CXXFLAGS) -O0 \
			-MM $$f -MT `echo $$f | sed 's%\.[^.]*%.o%'` \
			-isystem $(CLARISSE_INCL_DIR) >> $(DEPEND); \
	done

depend: $(DEPEND)

clean:
	$(RM) $(ALL_TARGETS) $(DEPEND)
	$(RM) $(LIBCLARISSE_STATIC)
	$(RM) $(LIBCLARISSE_SHARED)
	$(RM) $(LIBCLARISSE_OBJ_NAMES) $(LIBCLARISSE_CMA_NAMES)
	$(RM) $(CLARISSE_MODULE_OBJ_NAMES) $(CLARISSE_MODULE_SO_NAMES)

ifneq ($(strip $(wildcard $(DEPEND))),)
    include $(DEPEND)
endif

# Copyright (c) 2015 Double Negative Visual Effects
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
