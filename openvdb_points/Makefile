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
# Makefile for the OpenVDB Points library

# See INSTALL for a list of requirements.
#
# Targets:
#   lib                 the OpenVDB Points library
#
#   doc                 HTML documentation (doc/html/index.html)
#   pdfdoc              PDF documentation (doc/latex/refman.pdf;
#                       requires LaTeX and ghostscript)
#   vdb_test            unit tests for the OpenVDB library
#
#   all                 [default target] all of the above
#   install             install all of the above except vdb_test
#                       into subdirectories of DESTDIR
#   depend              recompute source file header dependencies
#   clean               delete generated files from the local directory
#   test                run tests
#
# Options:
#   abi=2               build for compatibility with the OpenVDB 2.x Grid ABI
#                       (some OpenVDB 3.x features will be disabled)
#   shared=no           link executables against static OpenVDB libraries
#                       (default: link against shared libraries)
#   debug=yes           build with debugging symbols and without optimization
#   verbose=yes         run commands (e.g., doxygen) in verbose mode


#
# The following variables must be defined, either here or on the command line
# (e.g., "make install DESTDIR=/usr/local"):
#
# Note that if you plan to build the Houdini OpenVDB tools (distributed
# separately), you must build the OpenVDB library and the Houdini tools
# against compatible versions of the Boost, OpenEXR and TBB libraries.
# Fortunately, all three are included in the Houdini HDK, so the relevant
# variables below point by default to the HDK library and header directories:
# $(HDSO) and $(HT)/include, respectively.  (Source the houdini_setup script
# to set those two environment variables.)
#

# The directory into which to install libraries, executables and header files
DESTDIR := /tmp/OpenVDB

# The parent directory of the openvdb/ header directory
OPENVDB_INCL_DIR := $(DESTDIR)/include
# The directory containing libopenvdb
OPENVDB_LIB_DIR := $(DESTDIR)/lib
OPENVDB_LIB := -lopenvdb

OPENVDB_POINTS_LIB := -lopenvdb_points

# The parent directory of the boost/ header directory
BOOST_INCL_DIR := $(HT)/include
# The directory containing libboost_iostreams, libboost_system, etc.
BOOST_LIB_DIR := $(HDSO)
BOOST_LIB := -lboost_iostreams -lboost_system
BOOST_THREAD_LIB := -lboost_thread

# The parent directory of the OpenEXR/ header directory
EXR_INCL_DIR := $(HT)/include
# The directory containing IlmImf
EXR_LIB_DIR := $(HDSO)
EXR_LIB := -lIlmImf

# The parent directory of the OpenEXR/ header directory (which contains half.h)
ILMBASE_INCL_DIR := $(EXR_INCL_DIR)
# The directory containing libIlmThread, libIlmThread, libHalf etc.
ILMBASE_LIB_DIR := $(EXR_LIB_DIR)
ILMBASE_LIB := -lIlmThread -lIex -lImath
HALF_LIB := -lHalf

# The parent directory of the tbb/ header directory
TBB_INCL_DIR := $(HT)/include
# The directory containing libtbb
TBB_LIB_DIR := $(HDSO)
TBB_LIB := -ltbb

# The parent directory of the blosc.h header
# (leave blank if Blosc is unavailable)
BLOSC_INCL_DIR := $(HT)/include
# The directory containing libblosc
BLOSC_LIB_DIR := $(HDSO)
BLOSC_LIB := -lblosc

# A scalable, concurrent malloc replacement library
# such as jemalloc (included in the Houdini HDK) or TBB malloc
# (leave blank if unavailable)
CONCURRENT_MALLOC_LIB := -ljemalloc
#CONCURRENT_MALLOC_LIB := -ltbbmalloc_proxy -ltbbmalloc
# The directory containing the malloc replacement library
CONCURRENT_MALLOC_LIB_DIR := $(HDSO)

# The parent directory of the cppunit/ header directory
# (leave blank if CppUnit is unavailable)
CPPUNIT_INCL_DIR := /rel/map/generic-2013.22/sys_include
# The directory containing libcppunit
CPPUNIT_LIB_DIR := /rel/depot/third_party_build/cppunit/1.10.2-7/opt-ws5-x86_64-gccWS5_64/lib
CPPUNIT_LIB := -lcppunit

# The parent directory of the log4cplus/ header directory
# (leave blank if log4cplus is unavailable)
LOG4CPLUS_INCL_DIR := /rel/folio/log4cplus/log4cplus-1.0.3-latest/sys_include
# The directory containing liblog4cplus
LOG4CPLUS_LIB_DIR := /rel/folio/log4cplus/log4cplus-1.0.3-latest/library
LOG4CPLUS_LIB := -llog4cplus

# The Doxygen executable
# (leave blank if Doxygen is unavailable)
DOXYGEN := doxygen


#
# Ideally, users shouldn't need to change anything below this line.
#

SHELL = /bin/bash

# Turn off implicit rules for speed.
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

ifeq (yes,$(strip $(debug)))
    OPTIMIZE := -g
else
    OPTIMIZE := -O3 -DNDEBUG
endif

ifeq (yes,$(strip $(verbose)))
    QUIET :=
    QUIET_TEST := -v
else
    QUIET := > /dev/null
    QUIET_TEST := $(QUIET)
endif

has_blosc := no
ifneq (,$(and $(BLOSC_LIB_DIR),$(BLOSC_INCL_DIR),$(BLOSC_LIB)))
    has_blosc := yes
endif

has_log4cplus := no
ifneq (,$(and $(LOG4CPLUS_LIB_DIR),$(LOG4CPLUS_INCL_DIR),$(LOG4CPLUS_LIB)))
    has_log4cplus := yes
endif

INCLDIRS := -I . -I .. -isystem $(OPENVDB_INCL_DIR) -isystem $(BOOST_INCL_DIR) -isystem $(ILMBASE_INCL_DIR) -isystem $(TBB_INCL_DIR)
ifeq (yes,$(has_blosc))
    INCLDIRS += -isystem $(BLOSC_INCL_DIR)
endif
ifeq (yes,$(has_log4cplus))
    INCLDIRS += -isystem $(LOG4CPLUS_INCL_DIR)
endif

CXXFLAGS += -pthread $(OPTIMIZE) $(INCLDIRS)
ifeq (yes,$(has_blosc))
    CXXFLAGS += -DOPENVDB_USE_BLOSC
endif
ifeq (yes,$(has_log4cplus))
    CXXFLAGS += -DOPENVDB_USE_LOG4CPLUS
endif
ifeq (2,$(strip $(abi)))
    CXXFLAGS += -DOPENVDB_2_ABI_COMPATIBLE
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
    -Wl,-rpath,$(OPENVDB_LIB_DIR) -L$(OPENVDB_LIB_DIR) $(OPENVDB_LIB) \
#
ifeq (yes,$(has_blosc))
    LIBS += -L$(BLOSC_LIB_DIR) $(BLOSC_LIB)
    LIBS_RPATH += -Wl,-rpath,$(BLOSC_LIB_DIR) -L$(BLOSC_LIB_DIR) $(BLOSC_LIB)
endif
ifeq (yes,$(has_log4cplus))
    LIBS += -L$(LOG4CPLUS_LIB_DIR) $(LOG4CPLUS_LIB)
    LIBS_RPATH += -Wl,-rpath,$(LOG4CPLUS_LIB_DIR) -L$(LOG4CPLUS_LIB_DIR) $(LOG4CPLUS_LIB)
endif
ifneq (,$(strip $(CONCURRENT_MALLOC_LIB)))
ifneq (,$(strip $(CONCURRENT_MALLOC_LIB_DIR)))
    LIBS_RPATH += -Wl,-rpath,$(CONCURRENT_MALLOC_LIB_DIR) -L$(CONCURRENT_MALLOC_LIB_DIR)
endif
endif
ifdef LINUX
    LIBS += -lrt
    LIBS_RPATH += -lrt
endif

INCLUDE_NAMES := \
    tools/AttributeArray.h \
    tools/PointDataGrid.h \
    openvdb.h \
#

SRC_NAMES := \
    tools/AttributeArray.cc \
    openvdb.cc \
#

UNITTEST_INCLUDE_NAMES := \
#

UNITTEST_SRC_NAMES := \
    unittest/main.cc \
    unittest/TestAttributeArray.cc \
    unittest/TestPointDataLeaf.cc \
#

DOC_FILES :=
DOC_INDEX := doc/html/index.html
DOC_PDF := doc/latex/refman.pdf


HEADER_SUBDIRS := $(dir $(INCLUDE_NAMES))

ALL_INCLUDE_FILES := \
    $(INCLUDE_NAMES) \
    $(UNITTEST_INCLUDE_NAMES) \
#
SRC_FILES := \
    $(SRC_NAMES) \
    $(UNITTEST_SRC_NAMES) \
#
ALL_SRC_FILES := $(SRC_FILES)

OBJ_NAMES := $(SRC_NAMES:.cc=.o)
UNITTEST_OBJ_NAMES := $(UNITTEST_SRC_NAMES:.cc=.o)

VERSION_H=$(OPENVDB_INCL_DIR)/openvdb/version.h

LIB_MAJOR_VERSION=$(shell grep 'define OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER ' \
    $(VERSION_H) | sed 's/[^0-9]*//g')
LIB_MINOR_VERSION=$(shell grep 'define OPENVDB_LIBRARY_MINOR_VERSION_NUMBER ' \
    $(VERSION_H) | sed 's/[^0-9]*//g')
LIB_PATCH_VERSION=$(shell grep 'define OPENVDB_LIBRARY_PATCH_VERSION_NUMBER ' \
    $(VERSION_H) | sed 's/[^0-9]*//g')

LIB_VERSION=$(LIB_MAJOR_VERSION).$(LIB_MINOR_VERSION).$(LIB_PATCH_VERSION)
SO_VERSION=$(LIB_MAJOR_VERSION).$(LIB_MINOR_VERSION)

LIBOPENVDBPOINTS_NAME=libopenvdb_points
LIBOPENVDBPOINTS_STATIC := $(LIBOPENVDBPOINTS_NAME).a
ifndef MBSD
LIBOPENVDBPOINTS_SHARED_NAME := $(LIBOPENVDBPOINTS_NAME).so
LIBOPENVDBPOINTS_SHARED := $(LIBOPENVDBPOINTS_NAME).so.$(LIB_VERSION)
LIBOPENVDBPOINTS_SONAME := $(LIBOPENVDBPOINTS_NAME).so.$(SO_VERSION)
LIBOPENVDBPOINTS_SONAME_FLAGS := -Wl,-soname,$(LIBOPENVDBPOINTS_SONAME)
else
LIBOPENVDBPOINTS_SHARED_NAME := $(LIBOPENVDBPOINTS_NAME).dylib
LIBOPENVDBPOINTS_SHARED := $(LIBOPENVDBPOINTS_NAME).$(LIB_VERSION).dylib
LIBOPENVDBPOINTS_SONAME := $(LIBOPENVDBPOINTS_NAME).$(SO_VERSION).dylib
LIBOPENVDBPOINTS_SONAME_FLAGS := -Wl,-install_name,$(DESTDIR)/lib/$(LIBOPENVDBPOINTS_SONAME)
endif

ifeq (no,$(strip $(shared)))
    LIBOPENVDBPOINTS := $(LIBOPENVDBPOINTS_STATIC)
else
    LIBOPENVDBPOINTS := $(LIBOPENVDBPOINTS_SHARED)
    LIBOPENVDBPOINTS_RPATH := -Wl,-rpath,$(DESTDIR)/lib
endif # shared

DEPEND := dependencies

# Get the list of dependencies that are newer than the current target,
# but limit the list to at most three entries.
list_deps = $(if $(wordlist 4,5,$(?F)),$(firstword $(?F)) and others,$(wordlist 1,3,$(?F)))

ALL_PRODUCTS := \
    $(LIBOPENVDBPOINTS) \
    vdb_test \
    $(DEPEND) \
    $(LIBOPENVDBPOINTS_SHARED_NAME) \
    $(LIBOPENVDBPOINTS_SONAME) \
#

.SUFFIXES: .o .cc

.PHONY: all clean depend doc install lib pdfdoc test

.cc.o:
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) -c $(CXXFLAGS) -fPIC -o $@ $<

all: lib vdb_test depend

$(OBJ_NAMES): %.o: %.cc
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) -c -DOPENVDB_PRIVATE $(CXXFLAGS) -fPIC -o $@ $<

ifneq (no,$(strip $(shared)))

# Build shared library
lib: $(LIBOPENVDBPOINTS_SHARED_NAME) $(LIBOPENVDBPOINTS_SONAME)

$(LIBOPENVDBPOINTS_SHARED_NAME): $(LIBOPENVDBPOINTS_SHARED)
	ln -f -s $< $@

$(LIBOPENVDBPOINTS_SONAME): $(LIBOPENVDBPOINTS_SHARED)
	ln -f -s $< $@

$(LIBOPENVDBPOINTS_SHARED): $(OBJ_NAMES)
	@echo "Building $@ because of $(list_deps)"
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LIBS_RPATH) $(LIBOPENVDBPOINTS_SONAME_FLAGS)

else

# Build static library
lib: $(LIBOPENVDBPOINTS)

$(LIBOPENVDBPOINTS_STATIC): $(OBJ_NAMES)
	@echo "Building $@ because of $(list_deps)"
	$(AR) cr $@ $^

endif # shared


$(DOC_INDEX): doxygen-config $(INCLUDE_NAMES) $(SRC_NAMES) $(DOC_FILES)
	@echo "Generating documentation because of $(list_deps)"
	echo 'OUTPUT_DIRECTORY=./doc' | cat doxygen-config - | $(DOXYGEN) - $(QUIET)

$(DOC_PDF): doxygen-config $(INCLUDE_NAMES) $(SRC_NAMES) $(DOC_FILES)
	@echo "Generating documentation because of $(list_deps)"
	echo -e 'OUTPUT_DIRECTORY=./doc\nGENERATE_LATEX=YES\nGENERATE_HTML=NO' \
	    | cat doxygen-config - | $(DOXYGEN) - $(QUIET) \
	    && cd ./doc/latex && make refman.pdf $(QUIET) \
	    && echo 'Created doc/latex/refman.pdf'

ifneq (,$(strip $(DOXYGEN)))
doc: $(DOC_INDEX)
pdfdoc: $(DOC_PDF)
else
doc:
	@echo "$@"': $$DOXYGEN is undefined'
pdfdoc:
	@echo "$@"': $$DOXYGEN is undefined'
endif


$(UNITTEST_OBJ_NAMES): %.o: %.cc
	@echo "Building $@ because of $(list_deps)"
	$(CXX) -c $(CXXFLAGS) -isystem $(CPPUNIT_INCL_DIR) -fPIC -o $@ $<

ifneq (,$(strip $(CPPUNIT_INCL_DIR)))
vdb_test: $(LIBOPENVDBPOINTS) $(UNITTEST_OBJ_NAMES)
	@echo "Building $@ because of $(list_deps)"
	$(CXX) $(CXXFLAGS) -o $@ $(UNITTEST_OBJ_NAMES) \
	    -Wl,-rpath,$(CPPUNIT_LIB_DIR) -L$(CPPUNIT_LIB_DIR) $(CPPUNIT_LIB) \
	    $(LIBOPENVDBPOINTS_RPATH) -L$(CURDIR) $(LIBOPENVDBPOINTS) \
	    $(LIBS_RPATH) $(CONCURRENT_MALLOC_LIB)

test: lib vdb_test
	@echo "Testing $(LIBOPENVDBPOINTS_NAME)"
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(CURDIR); ./vdb_test $(QUIET_TEST)
else
vdb_test:
	@echo "$@"': $$(CPPUNIT_INCL_DIR) is undefined'
test:
	@echo "$@"': $$(CPPUNIT_INCL_DIR) is undefined'
endif

install: lib doc
	mkdir -p $(DESTDIR)/include/openvdb_points
	@echo "Created $(DESTDIR)/include/openvdb_points"
	pushd $(DESTDIR)/include/openvdb_points > /dev/null; \
	    mkdir -p $(HEADER_SUBDIRS); popd > /dev/null
	for f in $(INCLUDE_NAMES); \
	    do cp -f $$f $(DESTDIR)/include/openvdb_points/$$f; done
	@echo "Copied header files to $(DESTDIR)/include"
	@#
	mkdir -p $(DESTDIR)/lib
	@echo "Created $(DESTDIR)/lib/"
	cp -f $(LIBOPENVDBPOINTS) $(DESTDIR)/lib
	pushd $(DESTDIR)/lib > /dev/null; \
	    if [ -f $(LIBOPENVDBPOINTS_SHARED) ]; then \
	        ln -f -s $(LIBOPENVDBPOINTS_SHARED) $(LIBOPENVDBPOINTS_SHARED_NAME); \
	        ln -f -s $(LIBOPENVDBPOINTS_SHARED) $(LIBOPENVDBPOINTS_SONAME); \
	    fi; \
	    popd > /dev/null
	@echo "Copied libopenvdb_points to $(DESTDIR)/lib/"
	if [ -d doc/html ]; \
	then \
	    mkdir -p $(DESTDIR)/share/doc/openvdb; \
	    echo "Created $(DESTDIR)/share/doc/openvdb/"; \
	    cp -r -f doc/html $(DESTDIR)/share/doc/openvdb; \
	    echo "Copied documentation to $(DESTDIR)/share/doc/openvdb/"; \
	fi

# TODO: This accumulates all source file dependencies into a single file
# containing a rule for each *.o file.  Consider generating a separate
# dependency file for each *.o file instead.
$(DEPEND): $(ALL_INCLUDE_FILES) $(ALL_SRC_FILES)
	@echo "Generating dependencies because of $(list_deps)"
	$(RM) $(DEPEND)
	for f in $(SRC_NAMES); \
	    do $(CXX) $(CXXFLAGS) -O0 \
	        -MM $$f -MT `echo $$f | sed 's%\.[^.]*%.o%'` >> $(DEPEND); \
	done
	if [ -d "$(CPPUNIT_INCL_DIR)" ]; \
	then \
	    for f in $(UNITTEST_SRC_NAMES); \
	        do $(CXX) $(CXXFLAGS) -O0 \
	            -MM $$f -MT `echo $$f | sed 's%\.[^.]*%.o%'` \
	            -isystem $(CPPUNIT_INCL_DIR) >> $(DEPEND); \
	    done; \
	fi

depend: $(DEPEND)

clean:
	$(RM) $(OBJ_NAMES) $(ALL_PRODUCTS) $(DEPEND)
	$(RM) $(LIBOPENVDBPOINTS_STATIC)
	$(RM) $(LIBOPENVDBPOINTS_SHARED)
	$(RM) $(UNITTEST_OBJ_NAMES)
	$(RM) -r ./doc/html ./doc/latex

ifneq (,$(strip $(wildcard $(DEPEND))))
    include $(DEPEND)
endif

# Copyright (c) 2012-2014 DreamWorks Animation LLC
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
