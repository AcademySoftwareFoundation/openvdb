# Copyright (c) 2012-2013 DreamWorks Animation LLC
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
# Makefile for the OpenVDB library

# See INSTALL for a list of requirements.
#
# Targets:
#   lib                 the OpenVDB library
#
#   doc                 HTML documentation (doc/html/index.html)
#   pdfdoc              PDF documentation (doc/latex/refman.pdf;
#                       requires LaTeX and ghostscript)
#   vdb_print           command-line tool to inspect OpenVDB files
#   vdb_render          command-line tool to ray-trace OpenVDB files
#   vdb_view            command-line tool to view OpenVDB files
#   vdb_test            unit tests for the OpenVDB library
#
#   all                 [default target] all of the above
#   install             install all of the above except vdb_test
#                       into subdirectories of INSTALL_DIR
#   depend              recompute source file header dependencies
#   clean               delete generated files from the local directory
#   test                run tests
#
# Options:
#   shared=no           link executables against static OpenVDB libraries
#                       (default: link against shared libraries)
#   debug=yes           build with debugging symbols and without optimization
#   verbose=yes         run commands (e.g., doxygen) in verbose mode

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
# Note that if you plan to build the Houdini OpenVDB tools (distributed
# separately), you must build the OpenVDB library and the Houdini tools
# against compatible versions of the Boost, OpenEXR and TBB libraries.
# Fortunately, all three are included in the Houdini HDK, so the relevant
# variables below point by default to the HDK library and header directories:
# $(HDSO) and $(HT)/include, respectively.  (Source the houdini_setup script
# to set those two environment variables.)
#

# The directory into which to install libraries, executables and header files
INSTALL_DIR := $(prefix)
include platform.mk

# The directory into which to install Houdini SOPs
HOUDINI_INSTALL_DIR := ./dso

# Default to $HFS if it already exists.
# The directory within which to source the houdini_setup script
ifneq (,$(HFS))
HOUDINI_DIR := $(HFS)
else
HOUDINI_DIR := /rel/third_party/houdini/12.5.376_64
endif

ifeq (,$(prefix))
  $(warning === WARNING === No prefix set)
endif
LOCAL_INCLUDE	:= $(prefix)/include
LOCAL_LIB	:= $(prefix)/lib

BOOST_VERSION	?= 1_51
# The parent directory of the boost/ header directory
BOOST_INCL_DIR	:= $(LOCAL_INCLUDE)/boost-$(BOOST_VERSION)

# The parent directory of the OpenEXR/ header directory
EXR_INCL_DIR := $(HOUDINI_DIR)/toolkit/include
# The directory containing libIlmImf, libIlmThread, etc.
EXR_LIB_DIR := $(HOUDINI_DIR)/dsolib
EXR_LIB := -lIlmImf -lIlmThread -lIex -lImath

# The parent directory of the OpenEXR/ header directory (which contains half.h)
HALF_INCL_DIR := $(LOCAL_INCLUDE)
# The directory containing libHalf
HALF_LIB_DIR := $(LOCAL_LIB)
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
TBB_INCL_DIR := $(LOCAL_INCLUDE)
# The directory containing libtbb
TBB_LIB_DIR := $(LOCAL_LIB)
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

# The parent directory of the zlib header directory (which contains zlib.h)
ZLIB_INCL_DIR := $(LOCAL_INCLUDE)
ZLIB_LIB_DIR := $(LOCAL_LIB)
ifdef WINDOWS_NT
  ifneq (full,$(strip $(debug)))
    ifeq (no,$(strip $(shared)))
      ZLIB_LIB := zlib.lib
    else
      ZLIB_LIB := zdll.lib
    endif
  else
    ifeq (no,$(strip $(shared)))
      ZLIB_LIB := zlib_d.lib
    else
      ZLIB_LIB := zdll_d.lib
    endif
  endif
else
  ZLIB_LIB := -lz
endif

# A scalable, concurrent malloc replacement library
# such as jemalloc (included in the Houdini HDK) or TBB malloc
# (leave blank if unavailable)
ifdef WINDOWS_NT
CONCURRENT_MALLOC_LIB :=
CONCURRENT_MALLOC_LIB_DIR :=
else
CONCURRENT_MALLOC_LIB := -ljemalloc
#CONCURRENT_MALLOC_LIB := -ltbbmalloc_proxy -ltbbmalloc
# The directory containing the malloc replacement library
CONCURRENT_MALLOC_LIB_DIR := $(HDSO)
endif

# The parent directory of the cppunit/ header directory
# (leave blank if CppUnit is unavailable)
CPPUNIT_INCL_DIR := $(LOCAL_INCLUDE)
# The directory containing libcppunit
CPPUNIT_LIB_DIR := $(LOCAL_LIB)
ifdef WINDOWS_NT
  ifneq (full,$(strip $(debug)))
    CPPUNIT_LIB := cppunit.lib
  else
    CPPUNIT_LIB := cppunitd.lib
  endif
else
  CPPUNIT_LIB := -lcppunit
endif

# The directory containing glfw.h
# (leave blank if GLFW is unavailable)
GLFW_INCL_DIR :=
# The directory containing libglfw
GLFW_LIB_DIR :=
GLFW_LIB :=

# The Doxygen executable
# (leave blank if Doxygen is unavailable)
DOXYGEN :=


#
# Ideally, users shouldn't need to change anything below this line.
#

SHELL = /bin/bash

PWD ?= $(abspath .)

# Turn off implicit rules for speed.
.SUFFIXES:

ifeq (full,$(strip $(debug)))
  OPTIMIZE := $(DEBUG_FLAGS)
  ifdef WINDOWS_NT
    LDFLAGS += -DEBUG
  else
  endif
else
  ifeq (yes,$(strip $(debug)))
    OPTIMIZE := $(DEBUG_FLAGS)
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

ifeq (yes,$(strip $(verbose)))
    QUIET :=
else
    QUIET := > /dev/null
endif

INCLDIRS := -I . -I .. -I $(BOOST_INCL_DIR) -I $(HALF_INCL_DIR) -I $(TBB_INCL_DIR) -I $(ZLIB_INCL_DIR)

CXXFLAGS += $(OPTIMIZE) $(INCLDIRS)

LIBS := \
    -L $(HALF_LIB_DIR) $(HALF_LIB) \
    -L $(TBB_LIB_DIR) $(TBB_LIB) \
    -L $(ZLIB_LIB_DIR) $(ZLIB_LIB) \
#
LIBS_RPATH := \
    -Wl,-rpath,'$$ORIGIN/.' $(LDDIROPT)$(HALF_LIB_DIR) $(HALF_LIB) \
    -Wl,-rpath,'$$ORIGIN/.' $(LDDIROPT)$(TBB_LIB_DIR) $(TBB_LIB) \
    -Wl,-rpath,'$$ORIGIN/.' $(LDDIROPT)$(ZLIB_LIB_DIR) $(ZLIB_LIB) \
#
ifeq (2,$(words $(strip $(CONCURRENT_MALLOC_LIB_DIR) $(CONCURRENT_MALLOC_LIB))))
ALLOC_LIBS_RPATH := \
    -Wl,-rpath,'$$ORIGIN/.' $(LDDIROPT)$(CONCURRENT_MALLOC_LIB_DIR) $(CONCURRENT_MALLOC_LIB)
else
ALLOC_LIBS_RPATH :=
endif

INCLUDE_NAMES := \
    Exceptions.h \
    Grid.h \
    io/Archive.h \
    io/Compression.h \
    io/File.h \
    io/GridDescriptor.h \
    io/Stream.h \
    math/BBox.h \
    math/Coord.h \
    math/FiniteDifference.h \
    math/Hermite.h \
    math/LegacyFrustum.h \
    math/Maps.h \
    math/Mat.h \
    math/Mat3.h \
    math/Mat4.h \
    math/Math.h \
    math/Operators.h \
    math/Proximity.h \
    math/QuantizedUnitVec.h \
    math/Quat.h \
    math/Ray.h \
    math/Stats.h \
    math/Stencils.h \
    math/Transform.h\
    math/Tuple.h\
    math/Vec2.h \
    math/Vec3.h \
    math/Vec4.h \
    Metadata.h \
    metadata/Metadata.h \
    metadata/MetaMap.h \
    metadata/StringMetadata.h \
    openvdb.h \
    Platform.h \
    PlatformConfig.h \
    tools/Composite.h \
    tools/Dense.h \
    tools/Filter.h \
    tools/GridOperators.h \
    tools/GridTransformer.h \
    tools/Interpolation.h \
    tools/LevelSetAdvect.h \
    tools/LevelSetFilter.h \
    tools/LevelSetFracture.h \
    tools/LevelSetRebuild.h \
    tools/LevelSetSphere.h \
    tools/LevelSetTracker.h \
    tools/LevelSetUtil.h \
    tools/MeshToVolume.h \
    tools/Morphology.h \
    tools/ParticlesToLevelSet.h \
    tools/PointAdvect.h \
    tools/PointScatter.h \
    tools/RayIntersector.h \
    tools/RayTracer.h \
    tools/Statistics.h \
    tools/ValueTransformer.h \
    tools/VolumeToMesh.h \
    tools/VolumeToSpheres.h \
    tree/InternalNode.h \
    tree/Iterator.h \
    tree/LeafManager.h \
    tree/LeafNode.h \
    tree/LeafNodeBool.h \
    tree/NodeUnion.h \
    tree/RootNode.h \
    tree/Tree.h \
    tree/TreeIterator.h \
    tree/Util.h \
    tree/ValueAccessor.h \
    Types.h \
    util/Formats.h \
    util/logging.h \
    util/MapsUtil.h \
    util/Name.h \
    util/NodeMasks.h \
    util/NullInterrupter.h \
    util/Util.h \
    version.h \
#

SRC_NAMES := \
    Grid.cc \
    io/Archive.cc \
    io/Compression.cc \
    io/File.cc \
    io/GridDescriptor.cc \
    io/Stream.cc \
    math/Hermite.cc \
    math/Maps.cc \
    math/Proximity.cc \
    math/QuantizedUnitVec.cc \
    math/Transform.cc \
    metadata/Metadata.cc \
    metadata/MetaMap.cc \
    openvdb.cc \
    Platform.cc \
    util/Formats.cc \
    util/Util.cc \
#

UNITTEST_INCLUDE_NAMES := \
    unittest/util.h \
#

UNITTEST_SRC_NAMES := \
    unittest/main.cc \
    unittest/TestBBox.cc \
    unittest/TestCoord.cc \
    unittest/TestCpt.cc \
    unittest/TestCurl.cc \
    unittest/TestDense.cc \
    unittest/TestDivergence.cc \
    unittest/TestDoubleMetadata.cc \
    unittest/TestExceptions.cc \
    unittest/TestFile.cc \
    unittest/TestFloatMetadata.cc \
    unittest/TestGradient.cc \
    unittest/TestGrid.cc \
    unittest/TestGridBbox.cc \
    unittest/TestGridDescriptor.cc \
    unittest/TestGridIO.cc \
    unittest/TestGridTransformer.cc \
    unittest/TestHermite.cc \
    unittest/TestInit.cc \
    unittest/TestInt32Metadata.cc \
    unittest/TestInt64Metadata.cc \
    unittest/TestInternalOrigin.cc \
    unittest/TestLaplacian.cc \
    unittest/TestLeaf.cc \
    unittest/TestLeafBool.cc \
    unittest/TestLeafIO.cc \
    unittest/TestLeafOrigin.cc \
    unittest/TestLevelSetRayIntersector.cc \
    unittest/TestLevelSetUtil.cc \
    unittest/TestLinearInterp.cc \
    unittest/TestMaps.cc \
    unittest/TestMat4Metadata.cc \
    unittest/TestMath.cc \
    unittest/TestMeanCurvature.cc \
    unittest/TestMeshToVolume.cc \
    unittest/TestMetadata.cc \
    unittest/TestMetadataIO.cc \
    unittest/TestMetaMap.cc \
    unittest/TestName.cc \
    unittest/TestNodeIterator.cc \
    unittest/TestNodeMask.cc \
    unittest/TestParticlesToLevelSet.cc \
    unittest/TestPrePostAPI.cc \
    unittest/TestQuadraticInterp.cc \
    unittest/TestQuantizedUnitVec.cc \
    unittest/TestQuat.cc \
    unittest/TestRay.cc \
    unittest/TestStats.cc \
    unittest/TestStream.cc \
    unittest/TestStringMetadata.cc \
    unittest/TestTools.cc \
    unittest/TestTransform.cc \
    unittest/TestTree.cc \
    unittest/TestTreeCombine.cc \
    unittest/TestTreeGetSetValues.cc \
    unittest/TestTreeIterators.cc \
    unittest/TestTreeVisitor.cc \
    unittest/TestValueAccessor.cc \
    unittest/TestVec2Metadata.cc \
    unittest/TestVec3Metadata.cc \
    unittest/TestVolumeRayIntersector.cc \
    unittest/TestVolumeToMesh.cc \
#

DOC_FILES := doc/doc.txt doc/faq.txt doc/changes.txt doc/codingstyle.txt doc/examplecode.txt doc/api_0_98_0.txt doc/math.txt
DOC_INDEX := doc/html/index.html
DOC_PDF := doc/latex/refman.pdf

LIBVIEWER_INCLUDE_NAMES := \
    viewer/Camera.h \
    viewer/ClipBox.h \
    viewer/Font.h \
    viewer/RenderModules.h \
    viewer/Viewer.h \
#
# Used for "install" target only
LIBVIEWER_PUBLIC_INCLUDE_NAMES := \
    viewer/Viewer.h \
#
LIBVIEWER_SRC_NAMES := \
    viewer/Camera.cc \
    viewer/ClipBox.cc \
    viewer/Font.cc \
    viewer/RenderModules.cc \
    viewer/Viewer.cc \
#
ifdef MBSD
    LIBVIEWER_FLAGS := -framework Cocoa -framework OpenGL -framework IOKit
else
    LIBVIEWER_FLAGS := -lGL -lGLU
endif


CMD_INCLUDE_NAMES := \
#

CMD_SRC_NAMES := \
    cmd/openvdb_print/main.cc \
#
RENDER_SRC_NAMES := \
    cmd/openvdb_render/main.cc \
#
VIEW_SRC_NAMES := \
    cmd/openvdb_view/main.cc \
#

HEADER_SUBDIRS := $(dir $(INCLUDE_NAMES))

ifdef WINDOWS_NT
GETOPT_INCLUDE := port/getopt.c
else
GETOPT_INCLUDE :=
endif

ALL_INCLUDE_FILES := \
    $(INCLUDE_NAMES) \
    $(UNITTEST_INCLUDE_NAMES) \
    $(CMD_INCLUDE_NAMES) \
    $(LIBVIEWER_INCLUDE_NAMES) \
    $(GETOPT_INCLUDE) \
#
SRC_FILES := \
    $(SRC_NAMES) \
    $(UNITTEST_SRC_NAMES) \
    $(CMD_SRC_NAMES) \
    $(RENDER_SRC_NAMES) \
    $(VIEW_SRC_NAMES) \
    $(LIBVIEWER_SRC_NAMES) \
#
ALL_SRC_FILES := $(SRC_FILES)

OBJ_NAMES := $(SRC_NAMES:.cc=.o)
CMD_OBJ_NAMES := $(CMD_SRC_NAMES:.cc=.o)
RENDER_OBJ_NAMES := $(RENDER_SRC_NAMES:.cc=.o)
VIEW_OBJ_NAMES := $(VIEW_SRC_NAMES:.cc=.o)
UNITTEST_OBJ_NAMES := $(UNITTEST_SRC_NAMES:.cc=.o)
LIBVIEWER_OBJ_NAMES := $(LIBVIEWER_SRC_NAMES:.cc=.o)

LIB_MAJOR_VERSION=$(shell grep 'OPENVDB_LIBRARY_MAJOR_VERSION = ' \
    version.h | sed 's/[^0-9]*//g')
LIB_MINOR_VERSION=$(shell grep 'OPENVDB_LIBRARY_MINOR_VERSION = ' \
    version.h | sed 's/[^0-9]*//g')
LIB_PATCH_VERSION=$(shell grep 'OPENVDB_LIBRARY_PATCH_VERSION = ' \
    version.h | sed 's/[^0-9]*//g')

LIB_VERSION=$(LIB_MAJOR_VERSION).$(LIB_MINOR_VERSION).$(LIB_PATCH_VERSION)

LIBOPENVDB_NAME=openvdb_sesi
LIBOPENVDB_STATIC := $(STATICLIB_PREFIX)$(LIBOPENVDB_NAME)$(STATICLIB_SUFFIX)
LIBOPENVDB_SHARED := $(SHAREDLIB_PREFIX)$(LIBOPENVDB_NAME)$(SHAREDLIB_SUFFIX)
LIBOPENVDB_SHARED_VERSIONED := $(LIBOPENVDB_SHARED).$(LIB_VERSION)
LIBOPENVDB_IMPORT := $(IMPORTLIB_PREFIX)$(LIBOPENVDB_NAME)$(IMPORTLIB_SUFFIX)

ifeq (no,$(strip $(shared)))
    LIBOPENVDB := $(LIBOPENVDB_STATIC)
    LIBOPENVDB_IMPORT := $(LIBOPENVDB_STATIC)
else

  ifeq (yes,$(strip $(VERSIONED_LIBS)))
    LIBOPENVDB := $(LIBOPENVDB_SHARED_VERSIONED)
  else
    LIBOPENVDB := $(LIBOPENVDB_SHARED)
  endif

    # For debugging, pass "rpath=yes" to encode the path to libopenvdb.so
    # into commands and SOPs.  With "rpath=no" (the default setting),
    # the directory containing libopenvdb.so will need to be added
    # to LD_LIBRARY_PATH.
    ifeq (yes,$(strip $(rpath)))
        LIBOPENVDB_RPATH := -Wl,-rpath,$(PWD) $(LDDIROPT)$(PWD) $(LIBOPENVDB)
    else
        LIBOPENVDB_RPATH := $(LDDIROPT)$(PWD) $(LIBOPENVDB)
    endif # rpath
endif # shared

CPPUNIT_RPATH := -Wl,-rpath,$(CPPUNIT_LIB_DIR) $(LDDIROPT)$(CPPUNIT_LIB_DIR) $(CPPUNIT_LIB)

ifdef WINDOWS_NT
    LIBS_RPATH := $(subst -L ,$(LDDIROPT),$(LIBS))
    LIBOPENVDB_RPATH := $(LIBOPENVDB_IMPORT)
    CPPUNIT_RPATH := $(LDDIROPT)$(CPPUNIT_LIB_DIR) $(CPPUNIT_LIB)
endif

DEPEND := dependencies

# Get the list of dependencies that are newer than the current target,
# but limit the list to at most three entries.
list_deps = $(if $(wordlist 4,5,$(?F)),$(firstword $(?F)) and others,$(wordlist 1,3,$(?F)))

ALL_TARGETS := \
    $(LIBOPENVDB_STATIC) \
    $(LIBOPENVDB_SHARED) \
    $(LIBOPENVDB_SHARED_VERSIONED) \
    $(LIBOPENVDB_IMPORT) \
    vdb_test$(EXE_SUFFIX) \
    vdb_test$(EXE_SUFFIX).manifest \
    vdb_test.{pdb,ilk} \
    vdb_print$(EXE_SUFFIX) \
    vdb_print$(EXE_SUFFIX).manifest \
    vdb_print.{pdb,ilk} \
    vdb_view$(EXE_SUFFIX) \
    vdb_view$(EXE_SUFFIX).manifest \
    vdb_view.{pdb,ilk} \
    $(DEPEND) \
#

.SUFFIXES: .o .cc

.PHONY: all clean depend install install_headers lib test sesi_archive 

.cc.o:
	$(call CompileCXX,)

all: lib vdb_print vdb_render vdb_test

$(OBJ_NAMES): %.o: %.cc
	$(call CompileCXX,-DOPENVDB_PRIVATE)

ifneq (no,$(strip $(shared)))

# Build shared library
lib: $(LIBOPENVDB_SHARED)

ifeq (yes,$(strip $(VERSIONED_LIBS)))
$(LIBOPENVDB_SHARED_VERSIONED): $(OBJ_NAMES)
	$(call BuildSharedLibrary,$(LIBS_RPATH))
$(LIBOPENVDB_SHARED): $(LIBOPENVDB_SHARED_VERSIONED)
	$(LN) $< $@
else
$(LIBOPENVDB_SHARED): $(OBJ_NAMES)
	$(call BuildSharedLibrary,$(LIBS_RPATH))
endif # versioned libs

ifdef WINDOWS_NT
$(LIBOPENVDB_IMPORT): lib
else
LIBOPENVDB_IMPORT :=
endif

else

# Build static library
lib: $(LIBOPENVDB_STATIC)

$(LIBOPENVDB_STATIC): $(OBJ_NAMES)
	@echo "Building $@ because of $(call list_deps)"
	$(AR) $(ARFLAGS) $(AROUTPUT)$@ $^

endif # shared

# BEGIN doc RULES
$(DOC_INDEX): doxygen-config $(INCLUDE_NAMES) $(SRC_NAMES) $(DOC_FILES)
	@echo "Generating documentation because of $(call list_deps)"
	echo 'OUTPUT_DIRECTORY=./doc' | cat doxygen-config - | $(DOXYGEN) - $(QUIET)

$(DOC_PDF): doxygen-config $(INCLUDE_NAMES) $(SRC_NAMES) $(DOC_FILES)
	@echo "Generating documentation because of $(call list_deps)"
	echo -e 'OUTPUT_DIRECTORY=./doc\nGENERATE_LATEX=YES\nGENERATE_HTML=NO' \
	    | cat doxygen-config - | $(DOXYGEN) - $(QUIET) \
	    && cd ./doc/latex && make refman.pdf $(QUIET) \
	    && echo 'Created doc/latex/refman.pdf'

ifneq ($(strip $(DOXYGEN)),)
doc: $(DOC_INDEX)
pdfdoc: $(DOC_PDF)
else
doc:
	@echo "$@"': $$DOXYGEN is undefined'
pdfdoc:
	@echo "$@"': $$DOXYGEN is undefined'
endif
# END doc RULES

# BEGIN vdb_print rules
vdb_print$(EXE_SUFFIX): $(LIBOPENVDB) $(CMD_OBJ_NAMES)
	$(call BuildExecutable, $(CMD_OBJ_NAMES) \
                                $(LIBS_RPATH) $(LIBOPENVDB_RPATH) \
				$(ALLOC_LIBS_RPATH))

ifneq (,$(strip $(EXE_SUFFIX)))
vdb_print: vdb_print$(EXE_SUFFIX)
endif
# END vdb_print rules

# BEGIN vdb_render rules
vdb_render$(EXE_SUFFIX): $(LIBOPENVDB) $(RENDER_OBJ_NAMES)
	$(call BuildExecutable, $(RENDER_OBJ_NAMES) \
                                $(LIBS_RPATH) $(LIBOPENVDB_RPATH) \
				$(ALLOC_LIBS_RPATH))

ifneq (,$(strip $(EXE_SUFFIX)))
vdb_render: vdb_render$(EXE_SUFFIX)
endif
# END vdb_render rules

# BEGIN vdb_view rules
ifneq (3,$(words $(strip $(GLFW_LIB_DIR) $(GLFW_INCL_DIR) $(GLFW_LIB))))
vdb_view:
	@echo "$@"': GLFW is unavailable'
else
# Create an openvdb_viewer/ symlink to the viewer/ subdirectory,
# to mirror the DWA directory structure.
openvdb_viewer:
	$(LN) viewer openvdb_viewer
$(LIBVIEWER_INCLUDE_NAMES): openvdb_viewer

$(LIBVIEWER_OBJ_NAMES): $(LIBVIEWER_INCLUDE_NAMES)
$(LIBVIEWER_OBJ_NAMES): %.o: %.cc
	$(call CompileCXX,-I $(GLFW_INCL_DIR) -DGL_GLEXT_PROTOTYPES=1)

$(LIBVIEWER_STATIC): $(LIBVIEWER_OBJ_NAMES) $(LIBOPENVDB)
	@echo "Building $@ because of $(call list_deps)"
	$(AR) cr $@ $^

$(LIBVIEWER_SHARED): $(LIBVIEWER_OBJ_NAMES) $(LIBOPENVDB)
	$(call BuildSharedLibrary,$(LIBS_RPATH))

ifneq (,$(strip $(EXE_SUFFIX)))
vdb_view: vdb_view$(EXE_SUFFIX)
endif

vdb_view$(EXE_SUFFIX): $(LIBOPENVDB) $(LIBVIEWER_OBJ_NAMES) $(VIEW_OBJ_NAMES)
	$(call BuildExecutable, $(VIEW_OBJ_NAMES) $(LIBVIEWER_OBJ_NAMES) \
		$(GLFW_RPATH) \
		$(LIBVIEWER_FLAGS) $(LIBS_RPATH) $(CONCURRENT_MALLOC_LIB) \
		$(LIBOPENVDB_RPATH) \
		$(ALLOC_LIBS_RPATH))
endif
# END vdb_view RULES

# BEGIN vdb_test RULES
ifneq (3,$(words $(strip $(CPPUNIT_LIB_DIR) $(CPPUNIT_INCL_DIR) $(CPPUNIT_LIB))))

$(UNITTEST_OBJ_NAMES): %.o: %.cc
	$(call CompileCXX,-I $(CPPUNIT_INCL_DIR))

vdb_test$(EXE_SUFFIX): $(LIBOPENVDB) $(UNITTEST_OBJ_NAMES)
	$(call BuildExecutable, $(UNITTEST_OBJ_NAMES) \
		$(CPPUNIT_RPATH) \
		$(LIBS_RPATH) $(LIBOPENVDB_RPATH) \
		$(ALLOC_LIBS_RPATH))

ifneq (,$(strip $(EXE_SUFFIX)))
vdb_test: vdb_test$(EXE_SUFFIX)
endif

ifdef MBSD
  APP_EXPORTS := export DYLD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(LOCAL_LIB):$(PWD)
else
  ifdef WINDOWS_NT
    APP_EXPORTS := true
  else
    APP_EXPORTS := export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(PWD)
  endif
endif

test: vdb_test
	$(APP_EXPORTS); ./vdb_test

else
vdb_test:
	@echo "$@"': $$(CPPUNIT_INCL_DIR) is undefined'
test:
	@echo "$@"': $$(CPPUNIT_INCL_DIR) is undefined'
endif
# END vdb_test RULES

install_headers: $(INCLUDE_NAMES)
	/bin/mkdir -p $(INSTALL_DIR)/include/openvdb
	@echo "Created $(INSTALL_DIR)/include/openvdb"
	pushd $(INSTALL_DIR)/include/openvdb > /dev/null; \
	    /bin/mkdir -p $(HEADER_SUBDIRS); popd > /dev/null
	for f in $(INCLUDE_NAMES); \
	    do /bin/cp -f $$f $(INSTALL_DIR)/include/openvdb/$$f; done
	@#
	@echo "Copied header files to $(INSTALL_DIR)/include"

install: install_headers lib vdb_print$(EXE_SUFFIX) vdb_render$(EXE_SUFFIX) vdb_view doc
	mkdir -p $(INSTALL_DIR)/lib
	@echo "Created $(INSTALL_DIR)/lib/"
	cp -f $(LIBOPENVDB) $(LIBOPENVDB_IMPORT) $(INSTALL_DIR)/lib
	pushd $(INSTALL_DIR)/lib > /dev/null; \
	    if [ -f $(LIBOPENVDB_SHARED_VERSIONED) ]; then \
		    $(LN) $(LIBOPENVDB_SHARED_VERSIONED) $(LIBOPENVDB_SHARED); fi; \
	    popd > /dev/null
	@echo "Copied libopenvdb to $(INSTALL_DIR)/lib/"
	@#
	mkdir -p $(INSTALL_DIR)/bin
	@echo "Created $(INSTALL_DIR)/bin/"
	cp -f vdb_print$(EXE_SUFFIX) $(INSTALL_DIR)/bin
	@echo "Copied vdb_print$(EXE_SUFFIX) to $(INSTALL_DIR)/bin/"
	cp -f vdb_render$(EXE_SUFFIX) $(INSTALL_DIR)/bin
	@echo "Copied vdb_render to $(INSTALL_DIR)/bin/"
	if [ -f vdb_view$(EXE_SUFFIX) ]; \
	then \
		cp -f vdb_view$(EXE_SUFFIX) $(INSTALL_DIR)/bin; \
		echo "Copied vdb_view to $(INSTALL_DIR)/bin/"; \
	fi
	@#
	if [ -d doc/html ]; \
	then \
	    mkdir -p $(INSTALL_DIR)/share/doc/openvdb; \
	    echo "Created $(INSTALL_DIR)/share/doc/openvdb/"; \
	    cp -r -f doc/html $(INSTALL_DIR)/share/doc/openvdb; \
	    echo "Copied documentation to $(INSTALL_DIR)/share/doc/openvdb/"; \
	fi

SESI_ARCHIVE_CONTENTS := \
    CHANGES \
    COPYRIGHT \
    INSTALL \
    LICENSE \
    README \
    Makefile \
    clmakedep.py \
    platform.mk \
    $(ALL_INCLUDE_FILES) \
    $(ALL_SRC_FILES) \
    $(DOC_FILES) \
    doxygen-config \
#
SESI_ARCHIVE_CONTENTS := $(addprefix openvdb/,$(SESI_ARCHIVE_CONTENTS))
SESI_ARCHIVE_NAME := openvdb-$(LIB_MAJOR_VERSION).$(LIB_MINOR_VERSION).$(LIB_PATCH_VERSION).tar.gz

sesi_archive:
	@echo "Creating $(SESI_ARCHIVE_NAME)"
	$(RM) $(SESI_ARCHIVE_NAME)
	$(LN) . openvdb
	/bin/tar cfz $(SESI_ARCHIVE_NAME) $(SESI_ARCHIVE_CONTENTS) || true
	$(RM) openvdb

ifdef WINDOWS_NT
HFS_DSOLIB := $(HFS)/bin
else
HFS_DSOLIB := $(HFS)/dsolib
endif

# Directly installs into $HFS for development builds
hfsinstall: $(LIBOPENVDB_SHARED)
	/bin/cp $(basename $(LIBOPENVDB_SHARED)).* $(HFS_DSOLIB)
hfsclean:
	$(RM) $(HFS_DSOLIB)/$(basename $(LIBOPENVDB_SHARED)).*

DEPFILES = $(call DepFromSrc,$(ALL_SRC_FILES))

# TODO: This accumulates all source file dependencies into a single file
# containing a rule for each *.o file.  Consider generating a separate
# dependency file for each *.o file instead.
$(DEPEND): $(ALL_INCLUDE_FILES) $(ALL_SRC_FILES)
	@echo "Generating dependencies because of $(call list_deps)"
	$(RM) $(DEPEND)
	for f in $(SRC_NAMES) $(LIBVIEWER_SRC_NAMES) $(CMD_SRC_NAMES) $(RENDER_SRC_NAMES) $(VIEW_SRC_NAMES); \
		do $(CXX) $(CXXFLAGS) -O0 \
			-MM $$f -MT `echo $$f | sed 's%\.[^.]*%.o%'` >> $(DEPEND); \
	done
	if [ -d "$(CPPUNIT_INCL_DIR)" ]; \
	then \
		for f in $(UNITTEST_SRC_NAMES); \
			do $(CXX) $(CXXFLAGS) -O0 \
				-MM $$f -MT `echo $$f | sed 's%\.[^.]*%.o%'` \
				-I $(CPPUNIT_INCL_DIR) >> $(DEPEND); \
		done; \
	fi

depend: $(DEPEND)

clean_obj:
	$(RM) $(OBJ_NAMES) $(UNITTEST_OBJ_NAMES) $(CMD_OBJ_NAMES) \
              $(LIBVIEWER_OBJ_NAMES) \
	      $(VIEW_OBJ_NAMES)

clean: clean_obj
	$(RM) $(ALL_TARGETS) $(DEPFILES)
	$(RM) $(LIBOPENVDB_STATIC)
	$(RM) $(LIBOPENVDB_SHARED_VERSIONED)
	$(RM) $(LIBOPENVDB_SHARED)
	$(RM) $(LIBOPENVDB_NAME).def
	$(RM) $(LIBOPENVDB_IMPORT)
	$(RM) *.lib *.exp *.ilk *.obj *.manifest *.pdb
	$(RM) -r ./doc/html
	$(RM) -r ./doc/html ./doc/latex

ifneq ($(DEPFILES),)
-include $(DEPFILES)
endif

# Copyright (c) 2012-2013 DreamWorks Animation LLC
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
