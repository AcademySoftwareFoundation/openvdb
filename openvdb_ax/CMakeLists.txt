# Copyright (c) 2015-2020 DNEG
#
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
#
# Redistributions of source code must retain the above copyright
# and license notice and the following restrictions and disclaimer.
#
# *     Neither the name of DNEG nor the names
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
#[=======================================================================[

  CMake Configuration for OpenVDB AX

#]=======================================================================]

# @note  Version 3.4 is required for re-generation of AX Grammar
cmake_minimum_required(VERSION 3.3)
project(OpenVDBAXCore LANGUAGES CXX)

option(OPENVDB_AX_SHARED "Build dynamically linked version of the ax library." ON)
option(OPENVDB_AX_STATIC "Build statically linked version of the ax library." ON)
option(USE_NEW_PASS_MANAGER [=[
Enable the new Optimization Pass Manager. This is a developer option.]=] OFF)

if(NOT OPENVDB_AX_SHARED AND NOT OPENVDB_AX_STATIC)
  message(FATAL_ERROR "Both static and shared ax OpenVDB libraries have been disabled. "
    "At least one must be enabled when building the ax library."
  )
endif()

mark_as_advanced(USE_NEW_PASS_MANAGER)

#########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "-------------- Configuring OpenVDBAX ---------------")
message(STATUS "----------------------------------------------------")

##########################################################################

# Configure grammar target

set(OPENVDB_AX_GRAMMAR_DIR ${CMAKE_CURRENT_BINARY_DIR}/openvdb_ax/grammar)
file(MAKE_DIRECTORY ${OPENVDB_AX_GRAMMAR_DIR})

if(OPENVDB_BUILD_AX_GRAMMAR)
  cmake_minimum_required(VERSION 3.4)
  find_package(FLEX REQUIRED)
  find_package(BISON 3.0 REQUIRED)

  # @note CMake lists don't seem to work with these options
  SET(BISON_COMPILE_FLAGS "${BISON_COMPILE_FLAGS} -Werror")

  if(OPENVDB_AX_GRAMMAR_NO_LINES)
    # Suppress #line directives
    SET(FLEX_COMPILE_FLAGS "${FLEX_COMPILE_FLAGS} -L")
    SET(BISON_COMPILE_FLAGS "${BISON_COMPILE_FLAGS} -l")
  endif()

  FLEX_TARGET(openvdb_ax_lexer grammar/axlexer.l ${OPENVDB_AX_GRAMMAR_DIR}/axlexer.cc
    COMPILE_FLAGS ${FLEX_COMPILE_FLAGS}
  )
  BISON_TARGET(openvdb_ax_parser grammar/axparser.y ${OPENVDB_AX_GRAMMAR_DIR}/axparser.cc
    DEFINES_FILE ${OPENVDB_AX_GRAMMAR_DIR}/axparser.h
    COMPILE_FLAGS ${BISON_COMPILE_FLAGS}
  )
  ADD_FLEX_BISON_DEPENDENCY(openvdb_ax_lexer openvdb_ax_parser)

  # Add a custom target so the language is only ever generated once
  add_custom_target(openvdb_ax_grammar
    COMMENT "Re-generate the AX language files."
    DEPENDS
      ${OPENVDB_AX_GRAMMAR_DIR}/axlexer.cc
      ${OPENVDB_AX_GRAMMAR_DIR}/axparser.cc
  )
else()
  file(COPY
    grammar/generated/axlexer.cc
    grammar/generated/axparser.cc
    grammar/generated/axparser.h
  DESTINATION
    ${OPENVDB_AX_GRAMMAR_DIR}
  )
endif()

#########################################################################

# Configure LLVM
set(MINIMUM_LLVM_VERSION 5.0.0)

find_package(LLVM REQUIRED)
find_library(found_LLVM LLVM HINTS ${LLVM_LIBRARY_DIRS})

if(found_LLVM)
  set(LLVM_LIBS "LLVM")
else()
  llvm_map_components_to_libnames(_llvm_libs
    native core executionengine support mcjit passes objcarcopts)
  set(LLVM_LIBS "${_llvm_libs}")
endif()

if(LLVM_PACKAGE_VERSION VERSION_LESS MINIMUM_LLVM_VERSION)
  message(FATAL_ERROR "Could NOT find LLVM: Found unsuitable version \"${LLVM_PACKAGE_VERSION}\", "
    "but required is at least \"${MINIMUM_LLVM_VERSION}\" (found ${LLVM_DIR})")
endif()

message(STATUS "Found LLVM: ${LLVM_DIR} (found suitable version \"${LLVM_PACKAGE_VERSION}\", "
  "minimum required is \"${MINIMUM_LLVM_VERSION}\")")

#########################################################################

# Configure other dependencies

find_package(Boost REQUIRED COMPONENTS random)

#########################################################################

set(OPENVDB_AX_LIBRARY_SOURCE_FILES
  ast/AST.cc
  ast/PrintTree.cc
  ast/Scanners.cc
  ${OPENVDB_AX_GRAMMAR_DIR}/axlexer.cc
  ${OPENVDB_AX_GRAMMAR_DIR}/axparser.cc
  codegen/ComputeGenerator.cc
  codegen/FunctionRegistry.cc
  codegen/FunctionTypes.cc
  codegen/PointComputeGenerator.cc
  codegen/PointFunctions.cc
  codegen/StandardFunctions.cc
  codegen/VolumeComputeGenerator.cc
  codegen/VolumeFunctions.cc
  compiler/Compiler.cc
  compiler/PointExecutable.cc
  compiler/VolumeExecutable.cc
  math/OpenSimplexNoise.cc
  )

set(OPENVDB_AX_AST_INCLUDE_FILES
  ast/AST.h
  ast/Literals.h
  ast/PrintTree.h
  ast/Scanners.h
  ast/Tokens.h
  ast/Visitor.h
)

set(OPENVDB_AX_CODEGEN_INCLUDE_FILES
  codegen/ComputeGenerator.h
  codegen/ConstantFolding.h
  codegen/FunctionRegistry.h
  codegen/Functions.h
  codegen/FunctionTypes.h
  codegen/PointComputeGenerator.h
  codegen/SymbolTable.h
  codegen/Types.h
  codegen/Utils.h
  codegen/VolumeComputeGenerator.h
  math/OpenSimplexNoise.h
)

set(OPENVDB_AX_COMPILER_INCLUDE_FILES
  compiler/Compiler.h
  compiler/CompilerOptions.h
  compiler/CustomData.h
  compiler/LeafLocalData.h
  compiler/PointExecutable.h
  compiler/AttributeRegistry.h
  compiler/VolumeExecutable.h
)

# @todo CMake >= 3.12, use an object library to consolidate shared/static
# builds. There are limitations with earlier versions of CMake when used with
# imported targets.

if(OPENVDB_AX_SHARED)
  add_library(openvdb_ax_shared SHARED ${OPENVDB_AX_LIBRARY_SOURCE_FILES})
  if(OPENVDB_BUILD_AX_GRAMMAR)
    add_dependencies(openvdb_ax_shared openvdb_ax_grammar)
  endif()
endif()

if(OPENVDB_AX_STATIC)
  add_library(openvdb_ax_static STATIC ${OPENVDB_AX_LIBRARY_SOURCE_FILES})
  if(OPENVDB_BUILD_AX_GRAMMAR)
    add_dependencies(openvdb_ax_static openvdb_ax_grammar)
  endif()
endif()

# Alias either the shared or static library to the generic OpenVDB
# target. Dependent components should use this target to build against
# such that they are always able to find a valid build of OpenVDB

if(OPENVDB_AX_SHARED)
  add_library(openvdb_ax ALIAS openvdb_ax_shared)
else()
  add_library(openvdb_ax ALIAS openvdb_ax_static)
endif()

set(OPENVDB_AX_CORE_DEPENDENT_LIBS
  OpenVDB::openvdb
  Boost::random
  ${LLVM_LIBS}
)

##########################################################################

# Configure static build

if(OPENVDB_AX_STATIC)
  target_compile_definitions(openvdb_ax_static PUBLIC ${LLVM_DEFINITIONS})
  if(NOT MATRIX_SUPPORT)
    target_compile_definitions(openvdb_ax_static PRIVATE -DOPENVDB_AX_NO_MATRIX)
  endif()

  target_include_directories(openvdb_ax_static
    PUBLIC ../
    PRIVATE . ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_include_directories(openvdb_ax_static
    SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS}
  )

  set_target_properties(openvdb_ax_static
    PROPERTIES OUTPUT_NAME openvdb_ax
  )

  target_link_libraries(openvdb_ax_static
    PUBLIC ${OPENVDB_AX_CORE_DEPENDENT_LIBS}
    )
  if(USE_NEW_PASS_MANAGER)
    target_compile_definitions(openvdb_ax_static PRIVATE -DUSE_NEW_PASS_MANAGER)
  endif()
endif()

if(OPENVDB_AX_SHARED)
  target_compile_definitions(openvdb_ax_shared PUBLIC ${LLVM_DEFINITIONS})
  if(NOT MATRIX_SUPPORT)
    target_compile_definitions(openvdb_ax_shared PRIVATE -DOPENVDB_AX_NO_MATRIX)
  endif()

  target_include_directories(openvdb_ax_shared
    PUBLIC ../
    PRIVATE . ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_include_directories(openvdb_ax_shared
    SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS}
  )

  # @TODO - Use a similar system to vdb to extract the version number from version.h

  set_target_properties(
    openvdb_ax_shared
    PROPERTIES
      OUTPUT_NAME openvdb_ax
      SOVERSION 0.2
      VERSION 0.2.0
    )

  if(OPENVDB_ENABLE_RPATH)
    # @todo There is probably a better way to do this for imported targets
    list(APPEND RPATHS
      ${Boost_LIBRARY_DIRS}
      ${IlmBase_LIBRARY_DIRS}
      ${OpenEXR_LIBRARY_DIRS}
      ${Log4cplus_LIBRARY_DIRS}
      ${Blosc_LIBRARY_DIRS}
      ${Tbb_LIBRARY_DIRS}
      ${LLVM_LIBRARY_DIRS}
    )
    list(REMOVE_DUPLICATES RPATHS)
    set_target_properties(openvdb_ax_shared
      PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
  endif()

  target_link_libraries(openvdb_ax_shared
    PUBLIC ${OPENVDB_AX_CORE_DEPENDENT_LIBS}
    )
  if(USE_NEW_PASS_MANAGER)
    target_compile_definitions(openvdb_ax_shared PRIVATE -DUSE_NEW_PASS_MANAGER)
  endif()
endif()

install(FILES Exceptions.h version.h DESTINATION include/openvdb_ax/)
install(FILES ${OPENVDB_AX_AST_INCLUDE_FILES} DESTINATION include/openvdb_ax/ast)
install(FILES ${OPENVDB_AX_CODEGEN_INCLUDE_FILES} DESTINATION include/openvdb_ax/codegen)
install(FILES ${OPENVDB_AX_COMPILER_INCLUDE_FILES} DESTINATION include/openvdb_ax/compiler)

if(OPENVDB_AX_STATIC)
  install(TARGETS openvdb_ax_static DESTINATION lib)
endif()

if(OPENVDB_AX_SHARED)
  install(TARGETS openvdb_ax_shared DESTINATION lib)
endif()
