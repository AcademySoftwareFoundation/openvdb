# Copyright (c) 2012-2017 DreamWorks Animation LLC
#
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
#
# Redistributions of source code must retain the above copyright
# and license notice and the following restrictions and disclaimer.
#
# *     Neither the name of DreamWorks Animation nor the names of
# its contributors may be used to endorse or promote products derived
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
# Makefile for OpenVDB Maya plugins
#

# Targets:
#   all                 [default target] all Maya plugins
#   install             install plugins, scripts, etc. into MAYA_DESTDIR
#   clean               delete generated files from the local directory
#   depend              recompute source file header dependencies
#
# Options:
#   abi=3               build for compatibility with the OpenVDB 3.x Grid ABI
#                       (some OpenVDB 4.x features will be disabled)
#   bits=32|64          build for 32-bit or 64-bit environments (default: 64)
#   debug=yes           build with debugging symbols and without optimization
#   verbose=yes         run commands in verbose mode


#
# The following variables must be defined, either here or on the command line
# (e.g., "make install DESTDIR=/usr/local"):
#
# Note that the OpenVDB library and these Maya plugins must be built
# against compatible versions of the Boost, OpenEXR and TBB libraries.
#

# MAYA_LOCATION should be set to the directory where Maya is installed.
ifeq (,$(MAYA_LOCATION))
$(warning Warning: $$(MAYA_LOCATION) is not set)
endif


# The directory into which to install libraries and header files
# (it is recommended that this match the DESTDIR in the OpenVDB
# library makefile)
DESTDIR := /tmp/OpenVDB

# The directory into which to install Maya plugins, scripts, etc.
MAYA_DESTDIR := $(DESTDIR)/maya

# The parent directory of the openvdb/ header directory
OPENVDB_INCL_DIR := $(DESTDIR)/include
# The directory containing libopenvdb
OPENVDB_LIB_DIR := $(DESTDIR)/lib
OPENVDB_LIB := -lopenvdb

# The parent directory of the boost/ header directory
BOOST_INCL_DIR :=
# The directory containing libboost_iostreams, libboost_system, etc.
BOOST_LIB_DIR :=
BOOST_LIB := -lboost_iostreams -lboost_system

# The parent directory of the OpenEXR/ header directory (which contains half.h)
ILMBASE_INCL_DIR :=
# The directory containing libHalf
ILMBASE_LIB_DIR :=
HALF_LIB := -lHalf

# The parent directory of the tbb/ header directory
TBB_INCL_DIR := $(MAYA_LOCATION)/include
# The directory containing libtbb
TBB_LIB_DIR := $(MAYA_LOCATION)/lib
TBB_LIB := -ltbb


#
# Ideally, users shouldn't need to change anything below this line.
#

SHELL = /bin/bash

# Turn off implicit rules for speed
.SUFFIXES:

ifneq (,$(INSTALL_DIR))
    $(warning Warning: $$(INSTALL_DIR) is no longer used; set $$(DESTDIR) instead.)
endif

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

MAYA_INCL_DIR := $(MAYA_LOCATION)/include

ifeq (yes,$(strip $(debug)))
    OPTIMIZE := -g
else
    OPTIMIZE := -O3 -DNDEBUG
endif

ifeq (yes,$(strip $(verbose)))
    QUIET :=
else
    QUIET := > /dev/null
endif

INCLDIRS := \
    -I . \
    -I $(OPENVDB_INCL_DIR) \
    -isystem $(BOOST_INCL_DIR) \
    -isystem $(ILMBASE_INCL_DIR) \
    -isystem $(TBB_INCL_DIR) \
    -isystem $(MAYA_INCL_DIR) \
#

CXXFLAGS += -std=c++11

CXXFLAGS += -pthread $(OPTIMIZE) $(INCLDIRS)
ifeq (3,$(strip $(abi)))
    CXXFLAGS += -DOPENVDB_3_ABI_COMPATIBLE
endif
# From $(MAYA_LOCATION)/devkit/plug-ins/buildconfig:
CXXFLAGS += \
    -D_BOOL \
    -DFUNCPROTO \
    -DGL_GLEXT_PROTOTYPES=1 \
    -DREQUIRE_IOSTREAM \
    -DUNIX \
    -fno-gnu-keywords \
    -fno-omit-frame-pointer \
    -fno-strict-aliasing \
    -funsigned-char \
    -Wno-comment \
    -Wno-multichar \
    -Wno-strict-aliasing \
#

ifeq (32,$(strip $(bits)))
    CXXFLAGS += -m32 -DBits32_
else
    CXXFLAGS += -m64 -DBits64_
endif

ifdef LINUX
    CXXFLAGS += -DLINUX
    ifeq (32,$(strip $(bits)))
        CXXFLAGS += -DLINUX_32
    else
        CXXFLAGS += -DLINUX_64
    endif
endif
ifdef MBSD
    # Darwin ld treats undefined symbols as errors by default;
    # change to runtime resolution, like Linux.
    CXXFLAGS += -undefined dynamic_lookup
endif

MAYA_LIBS := \
    -lOpenMayaFX \
    -lOpenMaya \
    -lFoundation \
#

LIBS := \
    -Wl,-rpath,$(OPENVDB_LIB_DIR) -L$(OPENVDB_LIB_DIR) $(OPENVDB_LIB) \
    -Wl,-rpath,$(ILMBASE_LIB_DIR) -L$(ILMBASE_LIB_DIR) $(HALF_LIB) \
    -Wl,-rpath,$(TBB_LIB_DIR) -L$(TBB_LIB_DIR) $(TBB_LIB) \
    -Wl,-rpath,$(BOOST_LIB_DIR) -L$(BOOST_LIB_DIR) $(BOOST_LIB) \
    -Wl,-rpath,$(MAYA_LOCATION)/lib -L$(MAYA_LOCATION)/lib $(MAYA_LIBS) \
    -ldl -lm -lz \
#
ifdef LINUX
    LIBS += -lrt
endif

MAYA_SRC_NAMES := \
    maya/OpenVDBCopyNode.cc \
    maya/OpenVDBData.cc \
    maya/OpenVDBFilterNode.cc \
    maya/OpenVDBFromMayaFluidNode.cc \
    maya/OpenVDBFromPolygonsNode.cc \
    maya/OpenVDBPlugin.cc \
    maya/OpenVDBReadNode.cc \
    maya/OpenVDBToPolygonsNode.cc \
    maya/OpenVDBTransformNode.cc \
    maya/OpenVDBUtil.cc \
    maya/OpenVDBVisualizeNode.cc \
    maya/OpenVDBWriteNode.cc \
#
MAYA_INCLUDE_NAMES := \
    maya/OpenVDBData.h \
    maya/OpenVDBUtil.h \
#

MAYA_SCRIPTS := \
    maya/AEOpenVDBCopyTemplate.mel \
    maya/AEOpenVDBFilterTemplate.mel \
    maya/AEOpenVDBFromMayaFluidTemplate.mel \
    maya/AEOpenVDBFromPolygonsTemplate.mel \
    maya/AEOpenVDBReadTemplate.mel \
    maya/AEOpenVDBToPolygonsTemplate.mel \
    maya/AEOpenVDBTransformTemplate.mel \
    maya/AEOpenVDBVisualizeTemplate.mel \
    maya/AEOpenVDBWriteTemplate.mel \
#

ALL_INCLUDE_FILES := \
    $(MAYA_INCLUDE_NAMES) \
#
MAYA_SRC_FILES := \
    $(MAYA_SRC_NAMES) \
#
ALL_SRC_FILES := $(MAYA_SRC_FILES)

MAYA_OBJ_NAMES := $(MAYA_SRC_NAMES:.cc=.o)
MAYA_SO_NAME := OpenVDB.so

DEPEND := maya_deps

# Get the list of dependencies that are newer than the current target,
# but limit the list to at most three entries.
list_deps = $(if $(wordlist 4,5,$(?F)),$(firstword $(?F)) and others,$(wordlist 1,3,$(?F)))

ALL_TARGETS := \
    $(DEPEND) \
    $(MAYA_SO_NAME) \
#

.SUFFIXES: .o .cc

.PHONY: all clean depend install

.cc.o:
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) -c $(CXXFLAGS) -fPIC -o $@ $<

all: $(ALL_TARGETS)

# Create an openvdb_maya/ symlink to the maya/ subdirectory,
# to mirror the DWA directory structure.
openvdb_maya:
	ln -f -s maya openvdb_maya
openvdb_maya/OpenVDBData.h openvdb_maya/OpenVDBUtil.h: openvdb_maya

$(MAYA_OBJ_NAMES): openvdb_maya/OpenVDBData.h openvdb_maya/OpenVDBUtil.h

$(MAYA_SO_NAME): $(MAYA_OBJ_NAMES)
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) $(CXXFLAGS) -fPIC -Wl,-Bsymbolic -shared -o $@ $^ $(LIBS)

install: $(MAYA_SO_NAME)
	mkdir -p $(MAYA_DESTDIR)/plugins
	@echo "Created $(MAYA_DESTDIR)/plugins"
	cp -f $(MAYA_SO_NAME) $(MAYA_DESTDIR)/plugins/
	@echo "Copied $(MAYA_SO_NAME) to $(MAYA_DESTDIR)/plugins/"
	mkdir -p $(MAYA_DESTDIR)/scripts
	@echo "Created $(MAYA_DESTDIR)/scripts"
	cp -f $(MAYA_SCRIPTS) $(MAYA_DESTDIR)/scripts/
	@echo "Copied MEL scripts to $(MAYA_DESTDIR)/scripts/"

# TODO: This accumulates all source file dependencies into a single file
# containing a rule for each *.o file.  Consider generating a separate
# dependency file for each *.o file instead.
$(DEPEND): $(ALL_INCLUDE_FILES) $(ALL_SRC_FILES)
	@echo "Generating dependencies because of $(call list_deps)"
	$(RM) $(DEPEND)
	for f in $(ALL_SRC_FILES); \
		do $(CXX) $(CXXFLAGS) -O0 -MM $$f -MT `echo $$f | sed 's%\.[^.]*%.o%'` \
			-isystem $(MAYA_INCL_DIR) >> $(DEPEND); \
	done

depend: $(DEPEND)

clean:
	$(RM) $(ALL_TARGETS) $(DEPEND) $(MAYA_OBJ_NAMES)

ifneq (,$(strip $(wildcard $(DEPEND))))
    include $(DEPEND)
endif

# Copyright (c) 2012-2017 DreamWorks Animation LLC
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
