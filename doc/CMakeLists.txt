# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Documentation

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
project(OpenVDBDocs LANGUAGES NONE)

include(GNUInstallDirs)

# If OPENVDB_DOXYGEN_HOUDINI is set, the Houdini documentation is also
# built. You can provide this module a value for Houdini_VERSION which will
# take into account any conditional compiled blocks. Whilst this doesn't
# require you to build the Houdini plugin, building from the root CMake
# with OPENVDB_BUILD_DOCS=ON and either OPENVDB_BUILD_HOUDINI_PLUGIN=ON or
# USE_HOUDINI=ON will define Houdini_VERSION. If Houdini_VERSION is not set,
# all documentation is included.
option(OPENVDB_DOXYGEN_HOUDINI "Build Houdini documentation" ON)
option(OPENVDB_DOXYGEN_AX "Build AX documentation" ON)

find_package(Doxygen REQUIRED doxygen)
if(MINIMUM_DOXYGEN_VERSION)
  if(DOXYGEN_VERSION VERSION_LESS MINIMUM_DOXYGEN_VERSION)
    message(WARNING "The doxygen-config doxyfile has been generated with version "
      "\"${MINIMUM_DOXYGEN_VERSION}\". Found Doxygen version \"${DOXYGEN_VERSION}\". "
      "Documentation may contain errors."
    )
  endif()
endif()

set(DOXY_FILES
  openvdb/openvdb
  openvdb/openvdb/io
  openvdb/openvdb/math
  openvdb/openvdb/points
  openvdb/openvdb/python/pyopenvdb.h
  openvdb/openvdb/tools
  openvdb/openvdb/tree
  openvdb/openvdb/util
  doc/build.txt
  doc/changes.txt
  doc/codingstyle.txt
  doc/dependencies.txt
  doc/doc.txt
  doc/examplecode.txt
  doc/faq.txt
  doc/math.txt
  doc/points.txt
  doc/python.txt)

set(DOXYGEN_PROJECT_NAME "OpenVDB")
set(DOXYGEN_PROJECT_NUMBER "${OpenVDB_VERSION}")
set(DOXYGEN_PROJECT_BRIEF "")
set(DOXYGEN_FILE_PATTERNS *.h *h.in) # headers only
set(DOXYGEN_EXTENSION_MAPPING .in=C) # parse CMake config headers as C
set(DOXYGEN_IMAGE_PATH "doc/img")
set(DOXYGEN_RECURSIVE NO)

set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_MAN NO)
set(DOXYGEN_GENERATE_LATEX NO)

set(DOXYGEN_HTML_COLORSTYLE_HUE 4)
set(DOXYGEN_HTML_COLORSTYLE_SAT 222)
set(DOXYGEN_HTML_TIMESTAMP NO) # for simpler website diffs

set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_STATIC YES)
set(DOXYGEN_EXTRACT_LOCAL_CLASSES NO)

set(DOXYGEN_ALPHABETICAL_INDEX NO)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_FULL_PATH_NAMES NO)
set(DOXYGEN_GENERATE_TODOLIST NO)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)
set(DOXYGEN_HIDE_SCOPE_NAMES YES)
set(DOXYGEN_INLINE_INHERITED_MEMB YES)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_CLASS_DIAGRAMS NO) # @todo use dot with class diagrams for relevant classes
set(DOXYGEN_ENABLED_SECTIONS "")

set(DOXYGEN_QUIET YES)

set(DOXYGEN_ALIASES
  [[ijk="(<i>i</i>,&nbsp;<i>j</i>,&nbsp;<i>k</i>)"]]
  [[xyz="(<i>x</i>,&nbsp;<i>y</i>,&nbsp;<i>z</i>)"]]
  [[const="<tt>const</tt>"]]
  "\"vdbnamespace=openvdb::v${OpenVDB_MAJOR_VERSION}_${OpenVDB_MINOR_VERSION}\""
  [["hunamespace=houdini_utils"]]
  [["hvdbnamespace=openvdb_houdini"]]
  # Use this command to create a link to an OpenVDB class, function, etc.
  # Usage is "@vdblink{<symbol>,<text>}", where <symbol> is a fully namespace-
  # qualified symbol minus the openvdb and version number namespace components
  # and <text> is the text of the link.
  # Example: @vdblink{tree::RootNode,root node}
  [[vdblink{2}="@link @vdbnamespace::\1 \2@endlink"]]
  # Deprecated; use the two-argument overload defined above.
  [[vdblink="@link @vdbnamespace"]]
  # Use these commands to create links to OpenVDB Houdini symbols.
  # Usage is "@hulink{<symbol>,<text>}", where <symbol> is the name of
  # a symbol in the houdini_utils namespace and <text> is the text of the link
  # (and likewise for @hvdblink).
  # Example: @hulink{ParmFactory::setDefault(),setDefault}
  # These stub implementations are used when Houdini support is disabled;
  # see doxygen-config-houdini for the full implementations.
  [[hulink{2}="<B>\2</B>"]]
  [[hvdblink{2}="<B>\2</B>"]]
  )

set(DOXYGEN_PREDEFINED
  "OPENVDB_VERSION_NAME=v${OpenVDB_MAJOR_VERSION}_${OpenVDB_MINOR_VERSION}"
  "OPENVDB_ABI_VERSION_NUMBER=${OpenVDB_MAJOR_VERSION}"
  [[__declspec(x):= __attribute__(x):=]]
  "OPENVDB_USE_LOG4CPLUS=")

if(OPENVDB_DOXYGEN_HOUDINI)
  # Append Houdini-specific settings to the Doxygen config file.
  if(Houdini_VERSION)
    # Extract the components of the Houdini version number into a list.
    string(REPLACE "." ";" HOUDINI_VERSION_COMPONENTS ${Houdini_VERSION})
    list(GET HOUDINI_VERSION_COMPONENTS 0 HVERS_MAJOR)
    list(GET HOUDINI_VERSION_COMPONENTS 1 HVERS_MINOR)
    list(GET HOUDINI_VERSION_COMPONENTS 2 HVERS_PATCH)
  else()
    set(HVERS_MAJOR 127)  # CMake treats (255 << 24) as a negative number
    set(HVERS_MINOR 0)
    set(HVERS_PATCH 0)
  endif()
  math(EXPR HVERS_INT "(((${HVERS_MAJOR} << 8) + ${HVERS_MINOR}) << 16) + ${HVERS_PATCH}")
  # Append Houdini version number macros to the Doxygen config file so that
  # the generated documentation reflects the API for that version
  # (i.e, taking into account any conditionally compiled blocks).
  list(APPEND DOXYGEN_PREDEFINED "UT_MAJOR_VERSION_INT=${HVERS_MAJOR}")
  list(APPEND DOXYGEN_PREDEFINED "UT_MINOR_VERSION_INT=${HVERS_MINOR}")
  list(APPEND DOXYGEN_PREDEFINED "UT_VERSION_INT=${HVERS_INT}")

  list(APPEND DOXYGEN_ALIASES
    [[hulink{2}="@link @hunamespace::\1 \2@endlink"]]
    [[hvdblink{2}="@link @hvdbnamespace::\1 \2@endlink"]]
    # Deprecated; use the two-argument overloads defined above.
    [[hulink="@link @hunamespace"]]
    [[hvdblink="@link @hvdbnamespace"]]
    )

  list(APPEND DOXYGEN_ENABLED_SECTIONS "OPENVDB_HOUDINI")

  list(APPEND DOXY_FILES
    doc/houdini.txt
    openvdb_houdini/openvdb_houdini/AttributeTransferUtil.h
    openvdb_houdini/openvdb_houdini/GEO_PrimVDB.h
    openvdb_houdini/openvdb_houdini/GT_GEOPrimCollectVDB.h
    openvdb_houdini/openvdb_houdini/GU_PrimVDB.h
    openvdb_houdini/openvdb_houdini/GU_VDBPointTools.h
    openvdb_houdini/openvdb_houdini/GeometryUtil.h
    openvdb_houdini/openvdb_houdini/ParmFactory.h
    openvdb_houdini/openvdb_houdini/PointUtils.h
    openvdb_houdini/openvdb_houdini/SOP_NodeVDB.h
    openvdb_houdini/openvdb_houdini/SOP_VDBVerbUtils.h
    openvdb_houdini/openvdb_houdini/UT_VDBTools.h
    openvdb_houdini/openvdb_houdini/UT_VDBUtils.h
    openvdb_houdini/openvdb_houdini/Utils.h)
endif()

if(OPENVDB_DOXYGEN_AX)
  set(DOXYGEN_EXTRA_PACKAGES amsmath) # for math formatting
  set(DOXYGEN_FORMULA_FONTSIZE 14)
  list(APPEND DOXYGEN_ENABLED_SECTIONS "OPENVDB_AX")

  list(APPEND DOXY_FILES
    doc/ax/ax.txt
    doc/ax/axcplusplus.txt
    doc/ax/axexamples.txt
    doc/ax/axfunctionlist.txt
    doc/ax/doc.txt
    openvdb_ax/openvdb_ax
    openvdb_ax/openvdb_ax/ast
    openvdb_ax/openvdb_ax/cmd
    openvdb_ax/openvdb_ax/codegen
    openvdb_ax/openvdb_ax/compiler
    openvdb_ax/openvdb_ax/grammar
    openvdb_ax/openvdb_ax/grammar/generated
    openvdb_ax/openvdb_ax/math
    openvdb_ax/openvdb_ax/test)
endif()

set(DOXYGEN_VERBATIM_VARS DOXYGEN_ALIASES DOXYGEN_PREDEFINED)

# Always add to build target (can't use USE_STAMP_FILE as we include
# complete directories in DOXY_FILES)
doxygen_add_docs(doc ${DOXY_FILES}
  ALL WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Generating API documentation with Doxygen")

# Suppress "Installing..." messages for all but one of the hundreds of generated files.
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/html/index.html DESTINATION ${CMAKE_INSTALL_DOCDIR}/html)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${CMAKE_INSTALL_DOCDIR} MESSAGE_NEVER)
