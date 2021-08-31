# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB AX

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
project(OpenVDBAXCore LANGUAGES CXX)

include(GNUInstallDirs)

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

if(NOT OPENVDB_BUILD_CORE)
  set(OPENVDB_LIB OpenVDB::openvdb)
else()
  set(OPENVDB_LIB openvdb)
endif()

# Configure grammar target

set(OPENVDB_AX_GRAMMAR_DIR ${CMAKE_CURRENT_BINARY_DIR}/openvdb_ax/grammar)
file(MAKE_DIRECTORY ${OPENVDB_AX_GRAMMAR_DIR})

if(OPENVDB_BUILD_AX_GRAMMAR)
  find_package(FLEX REQUIRED)
  find_package(BISON 3.0 REQUIRED)

  set(BISON_COMPILE_FLAGS "${BISON_COMPILE_FLAGS} -Werror")

  if(OPENVDB_AX_GRAMMAR_NO_LINES)
    # Suppress #line directives
    set(FLEX_COMPILE_FLAGS "${FLEX_COMPILE_FLAGS} -L")
    set(BISON_COMPILE_FLAGS "${BISON_COMPILE_FLAGS} -l")
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
      ${OPENVDB_AX_GRAMMAR_DIR}/axparser.cc)
endif()

#########################################################################

# Configure LLVM

find_package(LLVM REQUIRED)
find_library(found_LLVM LLVM HINTS ${LLVM_LIBRARY_DIRS})

if(found_LLVM)
  set(LLVM_LIBS "LLVM")
else()
  llvm_map_components_to_libnames(_llvm_libs
    native core executionengine support mcjit passes)
  set(LLVM_LIBS "${_llvm_libs}")
endif()

if(MINIMUM_LLVM_VERSION)
  if(LLVM_PACKAGE_VERSION VERSION_LESS MINIMUM_LLVM_VERSION)
    message(FATAL_ERROR "Could NOT find LLVM: Found unsuitable version \"${LLVM_PACKAGE_VERSION}\", "
      "but required is at least \"${MINIMUM_LLVM_VERSION}\" (found ${LLVM_DIR})")
  endif()
endif()
if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_LLVM_VERSION)
  if(LLVM_PACKAGE_VERSION VERSION_LESS FUTURE_MINIMUM_LLVM_VERSION)
    message(DEPRECATION "Support for LLVM versions < ${FUTURE_MINIMUM_LLVM_VERSION} "
      "is deprecated and will be removed.")
  endif()
endif()

message(STATUS "Found LLVM: ${LLVM_DIR} (found suitable version \"${LLVM_PACKAGE_VERSION}\", "
  "minimum required is \"${MINIMUM_LLVM_VERSION}\")")

#########################################################################

set(OPENVDB_AX_LIBRARY_SOURCE_FILES
  ast/Parse.cc
  ast/PrintTree.cc
  ast/Scanners.cc
  ax.cc
  codegen/ComputeGenerator.cc
  codegen/FunctionRegistry.cc
  codegen/FunctionTypes.cc
  codegen/PointComputeGenerator.cc
  codegen/PointFunctions.cc
  codegen/StandardFunctions.cc
  codegen/StringFunctions.cc
  codegen/Types.cc
  codegen/VolumeComputeGenerator.cc
  codegen/VolumeFunctions.cc
  compiler/Compiler.cc
  compiler/Logger.cc
  compiler/PointExecutable.cc
  compiler/VolumeExecutable.cc
  math/OpenSimplexNoise.cc
)

if(OPENVDB_BUILD_AX_GRAMMAR)
  list(APPEND OPENVDB_AX_LIBRARY_SOURCE_FILES
    ${OPENVDB_AX_GRAMMAR_DIR}/axlexer.cc
    ${OPENVDB_AX_GRAMMAR_DIR}/axparser.cc)
else()
  list(APPEND OPENVDB_AX_LIBRARY_SOURCE_FILES
    grammar/generated/axlexer.cc
    grammar/generated/axparser.cc)
endif()

set(OPENVDB_AX_AST_INCLUDE_FILES
  ast/AST.h
  ast/Parse.h
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
  codegen/PointLeafLocalData.h
  codegen/String.h
  codegen/SymbolTable.h
  codegen/Types.h
  codegen/Utils.h
  codegen/VolumeComputeGenerator.h
  math/OpenSimplexNoise.h
)

set(OPENVDB_AX_COMPILER_INCLUDE_FILES
  compiler/AttributeRegistry.h
  compiler/Compiler.h
  compiler/CompilerOptions.h
  compiler/CustomData.h
  compiler/Logger.h
  compiler/PointExecutable.h
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

# Configure shared settings

set(OPENVDB_AX_CORE_DEPENDENT_LIBS
  ${OPENVDB_LIB}
  ${LLVM_LIBS}
)

set(OPENVDB_AX_PUBLIC_DEFINES ${LLVM_DEFINITIONS})
set(OPENVDB_AX_PRIVATE_DEFINES "")
set(OPENVDB_AX_PUBLIC_OPTIONS "")

if(NOT LLVM_ENABLE_ASSERTIONS)
  # If no asserts in LLVM, build the address sanitiser build with -DNDEBUG
  # otherwise we get a missing symbol error from LLVM:
  #  undefined reference to `llvm::cfg::Update<llvm::BasicBlock*>::dump() const'
  list(APPEND OPENVDB_AX_PUBLIC_DEFINES
    "$<$<AND:$<CONFIG:ASAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:NDEBUG>")
endif()
if(NOT LLVM_ENABLE_RTTI)
  # If llvm has been built without rtti, disable the vptr sanitizer for AX
  list(APPEND OPENVDB_AX_PUBLIC_OPTIONS
    "$<$<AND:$<CONFIG:UBSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fno-sanitize=vptr>")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # TBB can't determine libstdc++ version if Clang is in use. This causes issues
  # with any TBB container using features from C++11 and greater (such as
  # tbb::concurrent_vector::emplace_back, used in the VolumeExecutable)
  # https://github.com/01org/tbb/issues/22
  # https://www.threadingbuildingblocks.org/docs/help/reference/appendices/known_issues/linux_os.html
  # https://www.threadingbuildingblocks.org/docs/help/reference/environment/feature_macros.html
  string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
  list(APPEND OPENVDB_AX_PUBLIC_DEFINES
    -DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
endif()

if(USE_NEW_PASS_MANAGER)
  list(APPEND OPENVDB_AX_PRIVATE_DEFINES -DUSE_NEW_PASS_MANAGER)
endif()

##########################################################################

# Configure static build

if(OPENVDB_AX_STATIC)
  target_compile_definitions(openvdb_ax_static PUBLIC ${OPENVDB_AX_PUBLIC_DEFINES})
  target_compile_definitions(openvdb_ax_static PRIVATE ${OPENVDB_AX_PRIVATE_DEFINES})
  target_compile_options(openvdb_ax_static PUBLIC ${OPENVDB_AX_PUBLIC_OPTIONS})
  target_include_directories(openvdb_ax_static PUBLIC ../)
  target_include_directories(openvdb_ax_static
    SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS}
  )

  if(OPENVDB_BUILD_AX_GRAMMAR)
    target_include_directories(openvdb_ax_static
      PRIVATE ${OPENVDB_AX_GRAMMAR_DIR}
    )
    target_compile_definitions(openvdb_ax_static
      PRIVATE -DOPENVDB_AX_REGENERATE_GRAMMAR
    )
  endif()

  set_target_properties(openvdb_ax_static
    PROPERTIES OUTPUT_NAME openvdb_ax
  )

  target_link_libraries(openvdb_ax_static PUBLIC ${OPENVDB_AX_CORE_DEPENDENT_LIBS})
endif()

# Configure shared build

if(OPENVDB_AX_SHARED)
  target_compile_definitions(openvdb_ax_shared PUBLIC ${OPENVDB_AX_PUBLIC_DEFINES})
  target_compile_definitions(openvdb_ax_shared PRIVATE ${OPENVDB_AX_PRIVATE_DEFINES})
  target_compile_options(openvdb_ax_shared PUBLIC ${OPENVDB_AX_PUBLIC_OPTIONS})
  target_include_directories(openvdb_ax_shared PUBLIC ../)
  target_include_directories(openvdb_ax_shared
    SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS}
  )

  if(OPENVDB_BUILD_AX_GRAMMAR)
    target_include_directories(openvdb_ax_shared
      PRIVATE ${OPENVDB_AX_GRAMMAR_DIR}
    )
    target_compile_definitions(openvdb_ax_shared
      PRIVATE -DOPENVDB_AX_REGENERATE_GRAMMAR
    )
  endif()

  set_target_properties(
    openvdb_ax_shared
    PROPERTIES
      OUTPUT_NAME openvdb_ax
      SOVERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}
      VERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}.${OpenVDB_PATCH_VERSION}
    )

  if(OPENVDB_ENABLE_RPATH)
    # @todo There is probably a better way to do this for imported targets
    list(APPEND RPATHS
      ${Boost_LIBRARY_DIRS}
      ${IlmBase_LIBRARY_DIRS}
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

  target_link_libraries(openvdb_ax_shared PUBLIC ${OPENVDB_AX_CORE_DEPENDENT_LIBS})
endif()

install(FILES ax.h Exceptions.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb_ax/)
install(FILES ${OPENVDB_AX_AST_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb_ax/ast)
install(FILES ${OPENVDB_AX_CODEGEN_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb_ax/codegen)
install(FILES ${OPENVDB_AX_COMPILER_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/openvdb_ax/compiler)

if(OPENVDB_AX_STATIC)
  install(TARGETS openvdb_ax_static
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

if(OPENVDB_AX_SHARED)
  install(TARGETS openvdb_ax_shared
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
