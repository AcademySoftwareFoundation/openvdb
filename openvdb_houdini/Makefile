# Copyright (c) 2012-2016 DreamWorks Animation LLC
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
#   houdinisops         Houdini OpenVDB SOPs (installed into HOUDINI_DESTDIR)
#
#   all                 [default target] all of the above
#   install             install houdinilib into subdirectories of DESTDIR,
#                       and the SOPs into HOUDINI_DESTDIR
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


# The directory into which to install libraries and header files
# (it's recommended that this match the DESTDIR in the OpenVDB
# library makefile)
DESTDIR := /tmp/OpenVDB

# The directory into which to install Houdini SOPs
HOUDINI_DESTDIR := $(HIH)/dso

# The parent directory of the openvdb/ header directory
OPENVDB_INCL_DIR := $(DESTDIR)/include
# The directory containing libopenvdb
OPENVDB_LIB_DIR := $(DESTDIR)/lib
OPENVDB_LIB := -lopenvdb

# The parent directory of the boost/ header directory
BOOST_INCL_DIR := $(HT)/include
# The directory containing libboost_iostreams, libboost_system, etc.
BOOST_LIB_DIR := $(HDSO)
BOOST_LIB := -lboost_iostreams -lboost_system

# The parent directory of the OpenEXR/ header directory (which contains half.h)
ILMBASE_INCL_DIR := $(HT)/include
# The directory containing libHalf
ILMBASE_LIB_DIR := $(HDSO)
HALF_LIB := -lHalf

# The parent directory of the tbb/ header directory
TBB_INCL_DIR := $(HT)/include
# The directory containing libtbb
TBB_LIB_DIR := $(HDSO)
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

HOUDINI_INCL_DIR := $(HT)/include
# Flags to be passed to the hcustom build script
HCUSTOM_FLAGS := -i $(HOUDINI_DESTDIR)
ifeq (yes,$(strip $(verbose)))
    HCUSTOM_FLAGS += -e
endif
# Extra flags to be passed through hcustom to the compiler or linker
HCUSTOM_EXTRA_CFLAGS := -DIGNORE_HDK_DEPRECATIONS
HCUSTOM_EXTRA_LDFLAGS :=

ifeq (yes,$(strip $(debug)))
    OPTIMIZE := -g
    HCUSTOM_FLAGS += -g
else
    OPTIMIZE := -O3 -DNDEBUG
endif

ifeq (yes,$(strip $(verbose)))
    QUIET :=
else
    QUIET := > /dev/null
endif

INCLDIRS := -I . -I .. -isystem $(BOOST_INCL_DIR) -isystem $(ILMBASE_INCL_DIR) -isystem $(TBB_INCL_DIR)
HCUSTOM_INCLDIRS := $(subst isystem,I,$(INCLDIRS))

CXXFLAGS += -pthread $(OPTIMIZE) $(INCLDIRS) -DIGNORE_HDK_DEPRECATIONS
ifeq (2,$(strip $(abi)))
    CXXFLAGS += -DOPENVDB_2_ABI_COMPATIBLE
    HCUSTOM_EXTRA_CFLAGS += -DOPENVDB_2_ABI_COMPATIBLE
endif
ifdef MBSD
    # Darwin ld treats undefined symbols as errors by default;
    # change to runtime resolution, like Linux.
    CXXFLAGS += -undefined dynamic_lookup
endif

ifneq (,$(strip $(wildcard $(HOUDINI_INCL_DIR))))
    HOUDINI_CXXFLAGS := $(shell hcustom -c)
    HOUDINI_LDFLAGS := $(shell hcustom -m)
endif

LIBS := \
    -ldl -lm -lz \
    -L$(ILMBASE_LIB_DIR) $(HALF_LIB) \
    -L$(TBB_LIB_DIR) $(TBB_LIB) \
    -L$(BOOST_LIB_DIR) $(BOOST_LIB) \
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

LIBHOUDINI_INCLUDE_NAMES := \
    houdini/AttributeTransferUtil.h \
    houdini/GeometryUtil.h \
    houdini/GEO_PrimVDB.h \
    houdini/GT_GEOPrimCollectVDB.h \
    houdini/GU_PrimVDB.h \
    houdini/GU_VDBPointTools.h \
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
    houdini/GT_GEOPrimCollectVDB.cc \
    houdini/GU_PrimVDB.cc \
    houdini/GU_VDBPointTools.cc \
    houdini/geometry.cc \
    houdini/ParmFactory.cc \
    houdini/SOP_NodeVDB.cc \
    houdini/UT_VDBUtils.cc \
    houdini/Utils.cc \
#
# Note that creation script names coincide with operator names,
# not (necessarily) with SOP names.
HOUDINI_SOP_SCRIPT_NAMES := \
    houdini/DW_OpenVDBRasterizePoints.cmd \
#
HOUDINI_SOP_SRC_NAMES := \
    houdini/SOP_OpenVDB_Advect.cc \
    houdini/SOP_OpenVDB_Advect_Points.cc \
    houdini/SOP_OpenVDB_Analysis.cc \
    houdini/SOP_OpenVDB_Clip.cc \
    houdini/SOP_OpenVDB_Combine.cc \
    houdini/SOP_OpenVDB_Convert.cc \
    houdini/SOP_OpenVDB_Create.cc \
    houdini/SOP_OpenVDB_Diagnostics.cc \
    houdini/SOP_OpenVDB_Fill.cc \
    houdini/SOP_OpenVDB_Filter.cc \
    houdini/SOP_OpenVDB_Filter_Level_Set.cc \
    houdini/SOP_OpenVDB_Fracture.cc \
    houdini/SOP_OpenVDB_From_Mask.cc \
    houdini/SOP_OpenVDB_From_Particles.cc \
    houdini/SOP_OpenVDB_From_Polygons.cc \
    houdini/SOP_OpenVDB_LOD.cc \
    houdini/SOP_OpenVDB_Metadata.cc \
    houdini/SOP_OpenVDB_Morph_Level_Set.cc \
    houdini/SOP_OpenVDB_Noise.cc \
    houdini/SOP_OpenVDB_Occlusion_Mask.cc \
    houdini/SOP_OpenVDB_Platonic.cc \
    houdini/SOP_OpenVDB_Prune.cc \
    houdini/SOP_OpenVDB_Rasterize_Points.cc \
    houdini/SOP_OpenVDB_Ray.cc \
    houdini/SOP_OpenVDB_Read.cc \
    houdini/SOP_OpenVDB_Rebuild_Level_Set.cc \
    houdini/SOP_OpenVDB_Remap.cc \
    houdini/SOP_OpenVDB_Remove_Divergence.cc \
    houdini/SOP_OpenVDB_Resample.cc \
    houdini/SOP_OpenVDB_Sample_Points.cc \
    houdini/SOP_OpenVDB_Sort_Points.cc \
    houdini/SOP_OpenVDB_Scatter.cc \
    houdini/SOP_OpenVDB_Segment.cc \
    houdini/SOP_OpenVDB_To_Polygons.cc \
    houdini/SOP_OpenVDB_To_Spheres.cc \
    houdini/SOP_OpenVDB_Topology_Mask.cc \
    houdini/SOP_OpenVDB_Transform.cc \
    houdini/SOP_OpenVDB_Vector_Merge.cc \
    houdini/SOP_OpenVDB_Vector_Split.cc \
    houdini/SOP_OpenVDB_Visualize.cc \
    houdini/SOP_OpenVDB_Write.cc \
#
HOUDINI_SOP_NAMES := $(basename $(notdir $(HOUDINI_SOP_SRC_NAMES)))

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
HOUDINI_SOP_SO_NAMES := $(addprefix $(HOUDINI_DESTDIR)/,\
    $(notdir $(HOUDINI_SOP_SRC_NAMES:.cc=.so)))

LIB_MAJOR_VERSION=$(shell grep 'define OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER ' \
    $(OPENVDB_INCL_DIR)/openvdb/version.h | sed 's/[^0-9]*//g')
LIB_MINOR_VERSION=$(shell grep 'define OPENVDB_LIBRARY_MINOR_VERSION_NUMBER ' \
    $(OPENVDB_INCL_DIR)/openvdb/version.h | sed 's/[^0-9]*//g')
LIB_PATCH_VERSION=$(shell grep 'define OPENVDB_LIBRARY_PATCH_VERSION_NUMBER ' \
    $(OPENVDB_INCL_DIR)/openvdb/version.h | sed 's/[^0-9]*//g')

LIB_VERSION=$(LIB_MAJOR_VERSION).$(LIB_MINOR_VERSION).$(LIB_PATCH_VERSION)

LIBHOUDINI_NAME=libopenvdb_houdini
LIBHOUDINI_STATIC := $(LIBHOUDINI_NAME).a
LIBHOUDINI_SHARED := $(LIBHOUDINI_NAME).so.$(LIB_VERSION)

ifeq (no,$(strip $(shared)))
    LIBHOUDINI := $(LIBHOUDINI_STATIC)
else
    LIBHOUDINI := $(LIBHOUDINI_SHARED)

    # For debugging, pass "rpath=yes" (the default setting) to encode
    # the path to libopenvdb.so into SOPs.  With "rpath=no", the directory
    # containing libopenvdb.so will need to be added to LD_LIBRARY_PATH.
    ifeq (no,$(strip $(rpath)))
        LIBOPENVDB_RPATH :=
    else
        LIBOPENVDB_RPATH := -Wl,-rpath,$(OPENVDB_LIB_DIR)
    endif # rpath
    HCUSTOM_EXTRA_LDFLAGS += $(LIBOPENVDB_RPATH)
endif # shared

# Environment variables to pass to hcustom
HCUSTOM_EXPORT :=
ifneq (,$(strip $(HCUSTOM_EXTRA_CFLAGS)$(HCUSTOM_EXTRA_LDFLAGS)))
    # Prior to Houdini 13.0.548 and 14.0.105, the only way to pass extra compiler flags
    # to hcustom was to append them to the compiler name and pass the result in the
    # environment variable HOUDINI_CC.
    ifeq (use_houdini_cc,$(shell test \
        \( $(HOUDINI_MAJOR_RELEASE) -lt 13 \) \
        -o \( \( $(HOUDINI_MAJOR_RELEASE) -eq 13 \) -a \( $(HOUDINI_BUILD_VERSION) -lt 548 \) \) \
        -o \( \( $(HOUDINI_MAJOR_RELEASE) -eq 14 \) -a \( $(HOUDINI_MINOR_RELEASE) -eq 0 \) \
            -a \( $(HOUDINI_BUILD_VERSION) -lt 105 \) \) \
        && echo use_houdini_cc))
        ifeq (,$(strip $(HOUDINI_CC)))
            # If HOUDINI_CC is not set, set it to CXX if CXX is set, otherwise set it to
            # something platform-appropriate.  (This replicates the logic in hcustom.)
            ifneq (,$(strip $(CXX)))
                HOUDINI_CC = $(CXX)
            else
                ifdef LINUX
                    HOUDINI_CC = g++
                else ifdef MBSD
                    HOUDINI_CC = clang++
                #else ifdef WINDOWS_NT  # TODO
                #    HOUDINI_CC =
                else
                    HOUDINI_CC =
                endif
            endif
        endif
        ifeq (,$(strip $(HOUDINI_CC)))
            $(error Cannot determine which C++ compiler to use with hcustom; please specify a compiler by setting $$(CXX))
        endif
        HCUSTOM_EXPORT := HOUDINI_CC='$(HOUDINI_CC) $(HCUSTOM_EXTRA_CFLAGS) $(HCUSTOM_EXTRA_LDFLAGS)'
    else
        # From Houdini 13.0.548 and 14.0.105 on, hcustom recognizes
        # the environment variables HCUSTOM_CFLAGS and HCUSTOM_LDFLAGS.
        ifneq (,$(strip $(HCUSTOM_EXTRA_CFLAGS)))
            HCUSTOM_EXPORT += HCUSTOM_CFLAGS=$(strip $(HCUSTOM_EXTRA_CFLAGS))
        endif
        ifneq (,$(strip $(HCUSTOM_EXTRA_LDFLAGS)))
            HCUSTOM_EXPORT += HCUSTOM_LDFLAGS=$(strip $(HCUSTOM_EXTRA_LDFLAGS))
        endif
    endif
endif
ifneq (,$(strip $(HCUSTOM_EXPORT)))
    HCUSTOM_EXPORT := export $(HCUSTOM_EXPORT);
endif

DEPEND := houdini_deps

# Get the list of dependencies that are newer than the current target,
# but limit the list to at most three entries.
list_deps = $(if $(wordlist 4,5,$(?F)),$(firstword $(?F)) and others,$(wordlist 1,3,$(?F)))

ALL_TARGETS := \
    $(LIBHOUDINI) \
    $(DEPEND) \
    $(LIBHOUDINI_NAME).so \
#

.SUFFIXES: .o .cc

.PHONY: all clean depend houdinisops houdinilib install

.cc.o:
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) -c $(CXXFLAGS) -fPIC -o $@ $<

all: depend houdinilib houdinisops

# Create an openvdb_houdini/ symlink to the houdini/ subdirectory,
# to mirror the DWA directory structure.
openvdb_houdini:
	ln -f -s houdini openvdb_houdini
openvdb_houdini/AttributeTransferUtil.h openvdb_houdini/GeometryUtil.h openvdb_houdini/GEO_PrimVDB.h openvdb_houdini/GT_GEOPrimCollectVDB.h openvdb_houdini/GU_PrimVDB.h openvdb_houdini/GU_VDBPointTools.h openvdb_houdini/SOP_NodeVDB.h openvdb_houdini/UT_VDBTools.h openvdb_houdini/UT_VDBUtils.h openvdb_houdini/Utils.h: openvdb_houdini

# Create a houdini_utils/ symlink to the houdini/ subdirectory,
# to mirror the DWA directory structure.
houdini_utils:
	ln -f -s houdini houdini_utils
houdini_utils/geometry.h houdini_utils/ParmFactory.h houdini_utils/OP_NodeChain.h: houdini_utils

$(LIBHOUDINI_OBJ_NAMES): houdini_utils/geometry.h houdini_utils/ParmFactory.h houdini_utils/OP_NodeChain.h openvdb_houdini/AttributeTransferUtil.h openvdb_houdini/GeometryUtil.h openvdb_houdini/GEO_PrimVDB.h openvdb_houdini/GT_GEOPrimCollectVDB.h openvdb_houdini/GU_PrimVDB.h openvdb_houdini/GU_VDBPointTools.h openvdb_houdini/SOP_NodeVDB.h openvdb_houdini/UT_VDBTools.h openvdb_houdini/UT_VDBUtils.h openvdb_houdini/Utils.h

$(LIBHOUDINI_OBJ_NAMES): %.o: %.cc
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) -c $(CXXFLAGS) $(HOUDINI_CXXFLAGS) \
	    -isystem $(HOUDINI_INCL_DIR) -fPIC -o $@ $<

$(LIBHOUDINI_STATIC): $(LIBHOUDINI_OBJ_NAMES)
	@echo "Building $@ because of $(call list_deps)"
	$(AR) cr $@ $^

$(LIBHOUDINI_SHARED): $(LIBHOUDINI_OBJ_NAMES)
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) $(CXXFLAGS) $(HOUDINI_LDFLAGS) -shared -o $@ $^ $(LIBS_RPATH)

$(LIBHOUDINI_NAME).so: $(LIBHOUDINI_SHARED)
	@# hcustom expects a library named "libname.so", not "libname.so.1.0".
	ln -f -s $< $@

houdinilib: $(LIBHOUDINI)
ifneq (no,$(strip $(shared)))
houdinilib: $(LIBHOUDINI_NAME).so
endif

# hcustom (see below) outputs both a .o and a .so file.
# The .o isn't dependent on $(LIBHOUDINI), but the .so is.
$(HOUDINI_SOP_OBJ_NAMES): $(LIBHOUDINI)

ifeq (no,$(strip $(shared)))
# Link SOPs against static OpenVDB libraries.
$(HOUDINI_SOP_OBJ_NAMES): %.o: %.cc
	@echo "Building $(HOUDINI_DESTDIR)/$(@:.o=.so) because of $(call list_deps)"
	$(HCUSTOM_EXPORT) $(HB)/hcustom $(HCUSTOM_FLAGS) $(HCUSTOM_INCLDIRS) $(LIBS) \
	    -L. -lopenvdb_houdini -L$(OPENVDB_LIB_DIR) $(OPENVDB_LIB) $< $(QUIET)
else
# Link SOPs against shared OpenVDB libraries.
$(HOUDINI_SOP_OBJ_NAMES): %.o: %.cc
	@echo "Building $(HOUDINI_DESTDIR)/$(@:.o=.so) because of $(call list_deps)"
	$(HCUSTOM_EXPORT) $(HB)/hcustom $(HCUSTOM_FLAGS) $(HCUSTOM_INCLDIRS) \
	    -L. -lopenvdb_houdini -L$(OPENVDB_LIB_DIR) $(OPENVDB_LIB) $< $(QUIET)
endif

$(HOUDINI_SOP_SO_NAMES): $(HOUDINI_DESTDIR)/%.so: houdini/%.o

houdinisops: $(HOUDINI_SOP_SO_NAMES) $(HOUDINI_SOP_SCRIPT_NAMES) houdini/SOP_OpenVDB.svg
	if [ -d "$(HOUDINI_DESTDIR)/../config/Icons" ]; \
	then \
	    cp -f houdini/SOP_OpenVDB.svg $(HOUDINI_DESTDIR)/../config/Icons/; \
	fi
	if [ ! -d "$(HOUDINI_DESTDIR)/../scripts/sop" ]; \
	then \
	    mkdir -p $(HOUDINI_DESTDIR)/../scripts/sop; \
	    echo "Created $(HOUDINI_DESTDIR)/../scripts/sop/"; \
	fi
	cp -f $(HOUDINI_SOP_SCRIPT_NAMES) $(HOUDINI_DESTDIR)/../scripts/sop/

install: houdinilib houdinisops
	mkdir -p $(DESTDIR)/include/openvdb_houdini
	@echo "Created $(DESTDIR)/include/openvdb_houdini"
	mkdir -p $(DESTDIR)/include/houdini_utils
	@echo "Created $(DESTDIR)/include/houdini_utils"
	cp -f $(LIBHOUDINI_INCLUDE_NAMES) $(DESTDIR)/include/openvdb_houdini/
	@# Move headers from openvdb_houdini/ to houdini_utils/.
	mv -f $(DESTDIR)/include/openvdb_houdini/geometry.h \
        $(DESTDIR)/include/openvdb_houdini/ParmFactory.h \
	    $(DESTDIR)/include/openvdb_houdini/OP_NodeChain.h \
	    $(DESTDIR)/include/houdini_utils/
	@echo "Copied header files to $(DESTDIR)/include"
	@#
	mkdir -p $(DESTDIR)/lib
	@echo "Created $(DESTDIR)/lib/"
	cp -f $(LIBHOUDINI) $(DESTDIR)/lib
	pushd $(DESTDIR)/lib > /dev/null; \
	    if [ -f $(LIBHOUDINI_SHARED) ]; then \
	        ln -f -s $(LIBHOUDINI_SHARED) $(LIBHOUDINI_NAME).so; fi; \
	    popd > /dev/null
	@echo "Copied libopenvdb_houdini to $(DESTDIR)/lib/"

# TODO: This accumulates all source file dependencies into a single file
# containing a rule for each *.o file.  Consider generating a separate
# dependency file for each *.o file instead.
$(DEPEND): $(ALL_INCLUDE_FILES) $(ALL_SRC_FILES)
	@echo "Generating dependencies because of $(call list_deps)"
	$(RM) $(DEPEND)
	for f in $(HOUDINI_SRC_FILES); \
	    do $(CXX) $(CXXFLAGS) $(HOUDINI_CXXFLAGS) -O0 \
	        -MM $$f -MT `echo $$f | sed 's%\.[^.]*%.o%'` \
	        -isystem $(HOUDINI_INCL_DIR) >> $(DEPEND); \
	done

depend: $(DEPEND)

clean:
	$(RM) $(ALL_TARGETS) $(DEPEND)
	$(RM) $(LIBHOUDINI_STATIC)
	$(RM) $(LIBHOUDINI_SHARED)
	$(RM) $(LIBHOUDINI_OBJ_NAMES)
	$(RM) $(HOUDINI_SOP_OBJ_NAMES) $(HOUDINI_SOP_SO_NAMES)
	$(RM) $(HOUDINI_DESTDIR)/../config/Icons/SOP_OpenVDB.svg
	$(RM) $(addprefix $(HOUDINI_DESTDIR)/../scripts/sop/,\
	    $(notdir $(HOUDINI_SOP_SCRIPT_NAMES)))

ifneq (,$(strip $(wildcard $(DEPEND))))
    include $(DEPEND)
endif

# Copyright (c) 2012-2016 DreamWorks Animation LLC
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
