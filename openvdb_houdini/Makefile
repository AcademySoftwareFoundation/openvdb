# Copyright (c) 2012-2014 DreamWorks Animation LLC
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
# Makefile for OpenVDB Houdini DSOs
#
# This makefile assumes that you are in a Houdini environment,
# i.e., that you have sourced the houdini_setup script.
#

# Targets:
#   houdinilib          utility library for Houdini OpenVDB SOPs
#   houdinisops         Houdini OpenVDB SOPs (installed into HOUDINI_INSTALL_DIR)
#
#   all                 [default target] all of the above
#   install             install houdinilib into subdirectories of INSTALL_DIR,
#                       and the SOPs into HOUDINI_INSTALL_DIR
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

# libConfig allows us to specify a single variable for whether we build
# debug/release and shared/static
ifdef libConfig
  ifeq (Debug,$(strip $(libConfig)))
    debug   := yes
    shared  := yes
  endif
  ifeq (DebugStatic,$(strip $(libConfig)))
    debug   := yes
    shared  := no
  endif
  ifeq (Release,$(strip $(libConfig)))
    debug   := no
    shared  := yes
  endif
  ifeq (ReleaseStatic,$(strip $(libConfig)))
    debug   := no
    shared  := no
  endif
  ifdef WINDOWS_NT
    ifeq (yes,$(strip $(debug)))
      debug := full
    endif
  endif
  $(info === Building $(libConfig): debug=$(debug) shared=$(shared) ===)
endif

#
# The following variables must be defined, either here or on the command line
# (e.g., "make install INSTALL_DIR=/usr/local"):
#
# Note that the OpenVDB library and these Houdini tools must be built
# against compatible versions of the Boost, OpenEXR and TBB libraries.
# Fortunately, all three are included in the Houdini HDK, so the relevant
# variables below point by default to the HDK library and header directories:
# $(HDSO) and $(HT)/include, respectively.  (Source the houdini_setup script
# to set those two environment variables.)
#

ifeq (,$(HFS))
$(warning Warning: $$(HFS) is not set; did you source houdini_setup?)
endif
ifdef WINDOWS_NT
HDSO := $(HFS)/custom/houdini/dsolib
endif

# The directory into which to install libraries and header files
# (it's recommended that this match the INSTALL_DIR in the OpenVDB
# library makefile)
INSTALL_DIR := $(prefix)
include ../openvdb/platform.mk

# The directory into which to install Houdini SOPs
HOUDINI_INSTALL_DIR := ./dso
ifeq (,$(prefix))
  $(warning === WARNING === No prefix set)
endif
LOCAL_INCLUDE	:= $(prefix)/include
LOCAL_LIB	:= $(prefix)/lib

# The parent directory of the openvdb/ header directory
OPENVDB_INCL_DIR := $(LOCAL_INCLUDE)
# The directory containing libopenvdb
OPENVDB_LIB_DIR := $(LOCAL_LIB)
ifdef WINDOWS_NT
  ifneq (full,$(strip $(debug)))
    ifeq (no,$(strip $(shared)))
      OPENVDB_LIB := openvdb_sesi.lib
    else
      OPENVDB_LIB := openvdb_sesi.lib
    endif
  else
    ifeq (no,$(strip $(shared)))
      OPENVDB_LIB := openvdb_sesi_d.lib
    else
      OPENVDB_LIB := openvdb_sesi_d.lib
    endif
  endif
else
  OPENVDB_LIB := -lopenvdb_sesi
endif

# The parent directory of the boost/ header directory
BOOST_INCL_DIR := $(LOCAL_INCLUDE)
# The directory containing libboost_iostreams, libboost_system, etc.
BOOST_LIB_DIR := $(LOCAL_LIB)
ifdef WINDOWS_NT
  BOOST_LIB :=
else
  BOOST_LIB := -lboost_iostreams -lboost_system
endif

# The parent directory of the OpenEXR/ header directory (which contains half.h)
HALF_INCL_DIR := $(HT)/include
# The directory containing libHalf
HALF_LIB_DIR := $(HDSO)
ifdef WINDOWS_NT
  ifneq (full,$(strip $(debug)))
    ifeq (no,$(strip $(shared)))
      HALF_LIB := libHalf.lib
    else
      HALF_LIB := Half.lib
    endif
  else
    ifeq (no,$(strip $(shared)))
      HALF_LIB := libHalf_d.lib
    else
      HALF_LIB := Half_d.lib
    endif
  endif
else
  HALF_LIB := -lHalf
endif

# The parent directory of the tbb/ header directory
TBB_INCL_DIR := $(HT)/include
# The directory containing libtbb
TBB_LIB_DIR := $(HDSO)
ifdef WINDOWS_NT
  ifneq (full,$(strip $(debug)))
    ifeq (no,$(strip $(shared)))
      TBB_LIB := libtbb.lib
    else
      TBB_LIB := tbb.lib
    endif
  else
    ifeq (no,$(strip $(shared)))
      TBB_LIB := libtbb_debug.lib
    else
      TBB_LIB := tbb_debug.lib
    endif
  endif
else
  TBB_LIB := -ltbb
endif


#
# Ideally, users shouldn't need to change anything below this line.
#

SHELL = /bin/bash

# Turn off implicit rules for speed
.SUFFIXES:

HOUDINI_INCL_DIR := $(HT)/include
# Flags to be passed to the hcustom build script
HCUSTOM_FLAGS := -i $(HOUDINI_INSTALL_DIR)
ifeq (yes,$(strip $(verbose)))
    HCUSTOM_FLAGS += -e
endif
# Extra flags to be passed through hcustom to the compiler or linker
HCUSTOM_EXTRA_FLAGS := -DIGNORE_HDK_DEPRECATIONS
# The compiler to be used by hcustom
ifeq ($(strip $(HOUDINI_CC)),)
    # If HOUDINI_CC is not set, set it to CXX if CXX is set, otherwise set it to "g++".
    # (This replicates the logic in hcustom.)
    ifneq ($(strip $(CXX)),)
        HOUDINI_CC = $(CXX)
    else
        HOUDINI_CC = g++
    endif
endif

ifeq (full,$(strip $(debug)))
  OPTIMIZE := $(DEBUG_FLAGS)
  ifdef WINDOWS_NT
    HCUSTOM_FLAGS += -G
    LDFLAGS += -DEBUG
  else
    HCUSTOM_FLAGS += -g
  endif
else
  ifeq (yes,$(strip $(debug)))
    OPTIMIZE := $(DEBUG_FLAGS)
    HCUSTOM_FLAGS += -g
    ifdef WINDOWS_NT
      LDFLAGS += -DEBUG
    endif
  else
    OPTIMIZE := $(OPTIMIZE_FLAGS)
  endif
endif
ifeq (,$(strip $(debug)))
  debug := no
endif
ifneq (no,$(strip $(debug)))
STATICLIB_SUFFIX := $(addprefix _d,$(STATICLIB_SUFFIX))
SHAREDLIB_SUFFIX := $(addprefix _d,$(SHAREDLIB_SUFFIX))
IMPORTLIB_SUFFIX := $(addprefix _d,$(IMPORTLIB_SUFFIX))
endif

HCUSTOM := $(HB)/hcustom $(HCUSTOM_FLAGS)

ifeq (yes,$(strip $(verbose)))
    QUIET :=
else
    QUIET := > /dev/null
endif

INCLDIRS := -I . -I .. $(CXXISYSTEM) $(BOOST_INCL_DIR) $(CXXISYSTEM) $(HALF_INCL_DIR) $(CXXISYSTEM) $(TBB_INCL_DIR)
HCUSTOM_INCLDIRS := $(subst isystem,I,$(INCLDIRS))

CXXFLAGS += $(OPTIMIZE) $(INCLDIRS) -DIGNORE_HDK_DEPRECATIONS
ifeq (2,$(strip $(abi)))
    CXXFLAGS += -DOPENVDB_2_ABI_COMPATIBLE
    HCUSTOM_EXTRA_FLAGS += -DOPENVDB_2_ABI_COMPATIBLE
endif
ifdef MBSD
    # Darwin ld treats undefined symbols as errors by default;
    # change to runtime resolution, like Linux.
    CXXFLAGS += -undefined dynamic_lookup
endif

ifneq ($(strip $(wildcard $(HOUDINI_INCL_DIR))),)
    HOUDINI_CXXFLAGS := $(shell hcustom -c)
    HOUDINI_LDFLAGS := $(shell hcustom -m)
endif

HCUSTOM_LIBS := \
    -L $(HALF_LIB_DIR) $(HALF_LIB) \
    -L $(TBB_LIB_DIR) $(TBB_LIB) \
    -L $(BOOST_LIB_DIR) $(BOOST_LIB) \
    -L $(OPENVDB_LIB_DIR) $(OPENVDB_LIB) \
#
LIBS_RPATH := \
    -Wl,-rpath,$(HALF_LIB_DIR) $(LDDIROPT)$(HALF_LIB_DIR) $(HALF_LIB) \
    -Wl,-rpath,$(TBB_LIB_DIR) $(LDDIROPT)$(TBB_LIB_DIR) $(TBB_LIB) \
    -Wl,-rpath,$(OPENVDB_LIB_DIR) $(LDDIROPT)$(OPENVDB_LIB_DIR) $(OPENVDB_LIB) \
    -Wl,-rpath,$(BOOST_LIB_DIR) -L$(BOOST_LIB_DIR) $(BOOST_LIB) \
#

LIBHOUDINI_INCLUDE_NAMES := \
    houdini/AttributeTransferUtil.h \
    houdini/GeometryUtil.h \
    houdini/GEO_PrimVDB.h \
    houdini/GT_GEOPrimCollectVDB.h \
    houdini/GU_PrimVDB.h \
    houdini/OP_NodeChain.h \
    houdini/geometry.h \
    houdini/ParmFactory.h \
    houdini/SOP_NodeVDB.h \
    houdini/UT_VDBTools.h \
    houdini/UT_VDBUtils.h \
    houdini/Utils.h \
#
LIBHOUDINI_SRC_NAMES := \
    houdini/GeometryUtil.cc \
    houdini/GEO_PrimVDB.cc \
    houdini/GEO_VDBTranslator.cc \
    houdini/GR_PrimVDBPoints.cc \
    houdini/GT_GEOPrimCollectVDB.cc \
    houdini/GU_PrimVDB.cc \
    houdini/geometry.cc \
    houdini/ParmFactory.cc \
    houdini/SOP_NodeVDB.cc \
    houdini/UT_VDBUtils.cc \
    houdini/Utils.cc \
#
HOUDINI_SOP_SRC_NAMES := \
    houdini/SOP_OpenVDB_Advect_Level_Set.cc \
    houdini/SOP_OpenVDB_Advect_Points.cc \
    houdini/SOP_OpenVDB_Analysis.cc \
    houdini/SOP_OpenVDB_Clip.cc \
    houdini/SOP_OpenVDB_Combine.cc \
    houdini/SOP_OpenVDB_Convert.cc \
    houdini/SOP_OpenVDB_Create.cc \
    houdini/SOP_OpenVDB_Fill.cc \
    houdini/SOP_OpenVDB_Filter.cc \
    houdini/SOP_OpenVDB_Filter_Level_Set.cc \
    houdini/SOP_OpenVDB_Fracture.cc \
    houdini/SOP_OpenVDB_From_Particles.cc \
    houdini/SOP_OpenVDB_From_Polygons.cc \
    houdini/SOP_OpenVDB_Metadata.cc \
    houdini/SOP_OpenVDB_Noise.cc \
    houdini/SOP_OpenVDB_Occlusion_Mask.cc \
    houdini/SOP_OpenVDB_Platonic.cc \
    houdini/SOP_OpenVDB_Prune.cc \
    houdini/SOP_OpenVDB_Ray.cc \
    houdini/SOP_OpenVDB_Read.cc \
    houdini/SOP_OpenVDB_Rebuild_Level_Set.cc \
    houdini/SOP_OpenVDB_Resample.cc \
    houdini/SOP_OpenVDB_Sample_Points.cc \
    houdini/SOP_OpenVDB_Scatter.cc \
    houdini/SOP_OpenVDB_To_Polygons.cc \
    houdini/SOP_OpenVDB_To_Spheres.cc \
    houdini/SOP_OpenVDB_Transform.cc \
    houdini/SOP_OpenVDB_Vector_Merge.cc \
    houdini/SOP_OpenVDB_Vector_Split.cc \
    houdini/SOP_OpenVDB_Visualize.cc \
    houdini/SOP_OpenVDB_Write.cc \
#
HOUDINI_SOP_NAMES := $(basename $(notdir $(HOUDINI_SOP_SRC_NAMES)))

#
#
# Replace filenames of the form "houdini/<name>.cc" with
# "$(DWA_HOUDINI_SOP_SRC_DIR)/<name>/<name>.cc"
DWA_HOUDINI_FILES += $(addprefix $(DWA_HOUDINI_SOP_SRC_DIR)/,\
    $(join $(HOUDINI_SOP_NAMES),$(patsubst houdini/%,/%,$(HOUDINI_SOP_SRC_NAMES))))

ALL_INCLUDE_FILES := \
    $(LIBHOUDINI_INCLUDE_NAMES) \
#
HOUDINI_SRC_FILES := \
    $(LIBHOUDINI_SRC_NAMES) \
    $(HOUDINI_SOP_SRC_NAMES) \
#
ALL_SRC_FILES := $(HOUDINI_SRC_FILES)

LIBHOUDINI_OBJ_NAMES := $(LIBHOUDINI_SRC_NAMES:.cc=.o)
HOUDINI_SOP_OBJ_NAMES := $(HOUDINI_SOP_SRC_NAMES:.cc=.o)
HOUDINI_SOP_SO_NAMES := $(addprefix $(HOUDINI_INSTALL_DIR)/,\
    $(notdir $(HOUDINI_SOP_SRC_NAMES:.cc=$(SHAREDLIB_SUFFIX))))

LIB_MAJOR_VERSION=$(shell grep 'define OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER ' \
    $(OPENVDB_INCL_DIR)/openvdb/version.h | sed 's/[^0-9]*//g')
LIB_MINOR_VERSION=$(shell grep 'define OPENVDB_LIBRARY_MINOR_VERSION_NUMBER ' \
    $(OPENVDB_INCL_DIR)/openvdb/version.h | sed 's/[^0-9]*//g')
LIB_PATCH_VERSION=$(shell grep 'define OPENVDB_LIBRARY_PATCH_VERSION_NUMBER ' \
    $(OPENVDB_INCL_DIR)/openvdb/version.h | sed 's/[^0-9]*//g')

LIB_VERSION=$(LIB_MAJOR_VERSION).$(LIB_MINOR_VERSION).$(LIB_PATCH_VERSION)

LIBHOUDINI_NAME=openvdb_houdini_sesi
LIBHOUDINI_STATIC := $(STATICLIB_PREFIX)$(LIBHOUDINI_NAME)$(STATICLIB_SUFFIX)
LIBHOUDINI_SHARED := $(SHAREDLIB_PREFIX)$(LIBHOUDINI_NAME)$(SHAREDLIB_SUFFIX)
LIBHOUDINI_SHARED_VERSIONED := $(LIBHOUDINI_SHARED).$(LIB_VERSION)
LIBHOUDINI_IMPORT := $(IMPORTLIB_PREFIX)$(LIBHOUDINI_NAME)$(IMPORTLIB_SUFFIX)

ifeq (no,$(strip $(shared)))
    LIBHOUDINI := $(LIBHOUDINI_STATIC)
    LIBHOUDINI_IMPORT := $(LIBHOUDINI_STATIC)
else ifdef WINDOWS_NT
    LIBHOUDINI := $(LIBHOUDINI_SHARED)
else
    LIBHOUDINI := $(LIBHOUDINI_SHARED_VERSIONED)
endif # shared

ifdef WINDOWS_NT
    LIBS_RPATH := $(subst -L ,$(LDDIROPT),$(HCUSTOM_LIBS))
endif

DEPEND := houdini_deps

# Get the list of dependencies that are newer than the current target,
# but limit the list to at most three entries.
list_deps = $(if $(wordlist 4,5,$(?F)),$(firstword $(?F)) and others,$(wordlist 1,3,$(?F)))

ALL_TARGETS := \
    $(LIBHOUDINI_STATIC) \
    $(LIBHOUDINI) \
    $(LIBHOUDINI_IMPORT) \
    $(DEPEND) \
#

.SUFFIXES: .o .cc

.PHONY: all clean depend houdinisops houdinilib install

.cc.o:
	$(call CompileCXX)

all: houdinilib houdinisops

# Create an openvdb_houdini/ symlink to the houdini/ subdirectory,
# to mirror the DWA directory structure.
openvdb_houdini:
	$(LN) houdini openvdb_houdini
openvdb_houdini/AttributeTransferUtil.h openvdb_houdini/GeometryUtil.h openvdb_houdini/GEO_PrimVDB.h openvdb_houdini/GT_GEOPrimCollectVDB.h openvdb_houdini/GU_PrimVDB.h openvdb_houdini/SOP_NodeVDB.h openvdb_houdini/UT_VDBTools.h openvdb_houdini/UT_VDBUtils.h openvdb_houdini/Utils.h: openvdb_houdini

# Create a houdini_utils/ symlink to the houdini/ subdirectory,
# to mirror the DWA directory structure.
houdini_utils:
	$(LN) houdini houdini_utils
houdini_utils/geometry.h houdini_utils/ParmFactory.h houdini_utils/OP_NodeChain.h: houdini_utils

$(LIBHOUDINI_OBJ_NAMES): houdini_utils/geometry.h houdini_utils/ParmFactory.h houdini_utils/OP_NodeChain.h openvdb_houdini/AttributeTransferUtil.h openvdb_houdini/GeometryUtil.h openvdb_houdini/GEO_PrimVDB.h openvdb_houdini/GT_GEOPrimCollectVDB.h openvdb_houdini/GU_PrimVDB.h openvdb_houdini/SOP_NodeVDB.h openvdb_houdini/UT_VDBTools.h openvdb_houdini/UT_VDBUtils.h openvdb_houdini/Utils.h

$(LIBHOUDINI_OBJ_NAMES): %.o: %.cc
	$(call CompileCXX,-DOPENVDB_HOUDINI_PRIVATE $(HOUDINI_CXXFLAGS))

$(LIBHOUDINI_STATIC): $(LIBHOUDINI_OBJ_NAMES)
	@echo "Building $@ because of $(call list_deps)"
	$(AR) $(ARFLAGS) $(AROUTPUT)$@ $^

$(LIBHOUDINI_SHARED_VERSIONED): $(LIBHOUDINI_OBJ_NAMES)
	$(call BuildSharedLibrary,$(HOUDINI_LDFLAGS) $(LIBS_RPATH))

ifdef WINDOWS_NT
$(LIBHOUDINI_SHARED): $(LIBHOUDINI_OBJ_NAMES)
	$(call BuildSharedLibrary,$(HOUDINI_LDFLAGS) $(LIBS_RPATH))
else
$(LIBHOUDINI_SHARED): $(LIBHOUDINI_SHARED_VERSIONED)
	@# hcustom expects a library named "libname.so", not "libname.so.1.0".
	$(LN) $< $@
endif

houdinilib: $(LIBHOUDINI)
ifneq (no,$(strip $(shared)))
houdinilib: $(LIBHOUDINI_SHARED)
endif

# hcustom (see below) outputs both a .o and a .so file.
# The .o isn't dependent on $(LIBHOUDINI), but the .so is.
$(HOUDINI_SOP_OBJ_NAMES): $(LIBHOUDINI)

ifndef WINDOWS_NT
HCUSTOM_LIBDEPS := -L . -l $(LIBHOUDINI_NAME) -L$(OPENVDB_LIB_DIR) $(OPENVDB_LIB)
else
HCUSTOM_LIBDEPS := -L . -l $(LIBHOUDINI_IMPORT) -L $(OPENVDB_LIB_DIR) -l $(OPENVDB_LIB)
endif

ifeq (no,$(strip $(shared)))
# Link SOPs against static OpenVDB libraries.
$(HOUDINI_SOP_OBJ_NAMES): %.o: %.cc
	@echo "Building $(HOUDINI_INSTALL_DIR)/$(@:.o=$(SHAREDLIB_SUFFIX)) because of $(call list_deps)"
	@# TODO: This seems to be the only way to pass extra compiler flags to hcustom.
	/bin/env HOUDINI_CC='$(HOUDINI_CC) $(HCUSTOM_EXTRA_FLAGS)' \
	    $(HCUSTOM) $(HOUDINI_INCLDIRS) \
		$(HCUSTOM_LIBS) $(HCUSTOM_LIBDEPS) $< $(QUIET)
else
# Link SOPs against shared OpenVDB libraries.
$(HOUDINI_SOP_OBJ_NAMES): %.o: %.cc
	@echo "Building $(HOUDINI_INSTALL_DIR)/$(@:.o=$(SHAREDLIB_SUFFIX)) because of $(call list_deps)"
	@# TODO: This seems to be the only way to pass extra compiler flags to hcustom.
	/bin/env HOUDINI_CC='$(HOUDINI_CC) $(HCUSTOM_EXTRA_FLAGS)' \
	    $(HCUSTOM) $(HOUDINI_INCLDIRS) $(HCUSTOM_LIBDEPS) $< $(QUIET)
endif

$(HOUDINI_SOP_SO_NAMES): $(HOUDINI_INSTALL_DIR)/%$(SHAREDLIB_SUFFIX): houdini/%.o

houdinisops: $(HOUDINI_SOP_SO_NAMES) houdini/SOP_OpenVDB.svg
	if [ -d "$(HOUDINI_INSTALL_DIR)/../config/Icons" ]; \
	then \
	    cp -f houdini/SOP_OpenVDB.svg $(HOUDINI_INSTALL_DIR)/../config/Icons/; \
	fi

install: houdinilib houdinisops
	mkdir -p $(INSTALL_DIR)/include/openvdb_houdini
	@echo "Created $(INSTALL_DIR)/include/openvdb_houdini"
	mkdir -p $(INSTALL_DIR)/include/houdini_utils
	@echo "Created $(INSTALL_DIR)/include/houdini_utils"
	cp -f $(LIBHOUDINI_INCLUDE_NAMES) $(INSTALL_DIR)/include/openvdb_houdini/
	@# Move headers from openvdb_houdini/ to houdini_utils/.
	mv -f $(INSTALL_DIR)/include/openvdb_houdini/geometry.h \
            $(INSTALL_DIR)/include/openvdb_houdini/ParmFactory.h \
	    $(INSTALL_DIR)/include/openvdb_houdini/OP_NodeChain.h \
	    $(INSTALL_DIR)/include/houdini_utils/
	@echo "Copied header files to $(INSTALL_DIR)/include"
	@#
	mkdir -p $(INSTALL_DIR)/lib
	@echo "Created $(INSTALL_DIR)/lib/"
	cp -f $(LIBHOUDINI) $(LIBHOUDINI_IMPORT) $(INSTALL_DIR)/lib
	pushd $(INSTALL_DIR)/lib > /dev/null; \
	    if [ -f $(LIBHOUDINI_SHARED) ]; then \
	        $(LN) $(LIBHOUDINI_SHARED_VERSIONED) $(LIBHOUDINI_SHARED); fi; \
	    popd > /dev/null
	@echo "Copied libopenvdb_houdini to $(INSTALL_DIR)/lib/"

ifdef WINDOWS_NT
HFS_DSOLIB := $(HFS)/bin
else
HFS_DSOLIB := $(HFS)/dsolib
endif

# Directly installs into $HFS for development builds
hfsinstall: $(LIBHOUDINI_SHARED) $(HOUDINI_SOP_SO_NAMES)
	   /bin/cp $(basename $(LIBHOUDINI_SHARED)).* $(HFS_DSOLIB) \
	&& /bin/cp $(addprefix $(HOUDINI_INSTALL_DIR)/, \
			$(addsuffix .*,$(HOUDINI_SOP_NAMES))) \
		   $(HFS)/houdini/dso \
	$(NULL)
hfsclean:
	   $(RM) $(HFS)/houdini/dso/$(basename $(LIBHOUDINI_SHARED)).* \
	&& $(RM) $(HFS)/houdini/dso/SOP_* \
	$(NULL)


DEPFILES = $(call DepFromSrc,$(ALL_SRC_FILES))

# TODO: This accumulates all source file dependencies into a single file
# containing a rule for each *.o file.  Consider generating a separate
# dependency file for each *.o file instead.
$(DEPEND): $(ALL_INCLUDE_FILES) $(ALL_SRC_FILES)
	@echo "Generating dependencies because of $(call list_deps)"
	$(RM) $(DEPEND)
	for f in $(HOUDINI_SRC_FILES); \
		do $(CXX) $(CXXFLAGS) $(HOUDINI_CXXFLAGS) -O0 \
			-MM $$f -MT `echo $$f | sed 's%\.[^.]*%.o%'` \
			-I $(HOUDINI_INCL_DIR) >> $(DEPEND); \
	done

depend: $(DEPEND)

clean:
	$(RM) $(ALL_TARGETS) $(DEPEND)
	$(RM) $(LIBHOUDINI_STATIC)
	$(RM) $(LIBHOUDINI_SHARED)
	$(RM) $(LIBHOUDINI_OBJ_NAMES)
	$(RM) $(LIBHOUDINI_NAME).def
	$(RM) $(HOUDINI_SOP_OBJ_NAMES) $(HOUDINI_SOP_SO_NAMES)
	$(RM) *.lib *.exp *.ilk *.obj *.manifest *.pdb
	$(RM) houdini/.*.d
	$(RM) -r  openvdb_houdini houdini_utils
	$(RM) $(HOUDINI_INSTALL_DIR)/../config/Icons/SOP_OpenVDB.svg

ifneq ($(DEPFILES),)
-include $(DEPFILES)
endif

# Copyright (c) 2012-2014 DreamWorks Animation LLC
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
