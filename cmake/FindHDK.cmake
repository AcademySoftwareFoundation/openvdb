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

#-*-cmake-*-
# - Find HDK
#
# Author : Nicholas Yue yue.nicholas@gmail.com
#
# This auxiliary CMake file helps in find the HDK headers and libraries
#
# HDK_FOUND            set if HDK is found.
# HDK_INCLUDE_DIR      HDK's include directory
# HDK_LIBRARY_DIR      HDK's library directory (Useful for cmake packaging to ship runtime libraries)
# Hdk_SDK_LIBRARIES    HDK libraries (as a collection for easier reference)

FIND_PACKAGE ( PackageHandleStandardArgs )

GET_FILENAME_COMPONENT ( HDK_PACKAGE_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
# MESSAGE ( "CMAKE_CURRENT_LIST_FILE ${CMAKE_CURRENT_LIST_FILE}" )
# MESSAGE ( "HDK_PACKAGE_DIR ${HDK_PACKAGE_DIR}" )

SET ( HDK_VERSION_IN_SYS_SYS_VERSION_H ON )
OPTION ( HDK_AUTO_GENERATE_SESITAG "Automatically generate <Target>_sesitag.C and add to SOP being built" OFF )

# Houdini 15 and above defines version in SYS/SYS_Version.h
SET ( HDK_VERSION_FILE_PATH "toolkit/include/SYS/SYS_Version.h" )
FIND_PATH( HDK_LOCATION ${HDK_VERSION_FILE_PATH}
  "$ENV{HFS}"
  NO_DEFAULT_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  )

# MESSAGE ( "NICHOLAS 0000" )

# Fall back for Houdini version less than 15.0
IF (NOT HDK_LOCATION)
  # MESSAGE ( "NICHOLAS 0100" )
  SET ( HDK_VERSION_FILE_PATH "toolkit/include/UT/UT_Version.h" )
  # MESSAGE ( "NICHOLAS 0200 HDK_VERSION_FILE_PATH = ${HDK_VERSION_FILE_PATH}" )
  FIND_PATH( HDK_LOCATION ${HDK_VERSION_FILE_PATH}
	"$ENV{HFS}"
	NO_DEFAULT_PATH
	NO_SYSTEM_ENVIRONMENT_PATH
	)
  # MESSAGE ( "NICHOLAS 0300 HDK_LOCATION = ${HDK_LOCATION}" )
  SET ( HDK_VERSION_IN_SYS_SYS_VERSION_H OFF )
ENDIF ()

SET ( HDK_VERSION_FILE ${HDK_LOCATION}/${HDK_VERSION_FILE_PATH} )
# MESSAGE ( "HDK_VERSION_FILE = ${HDK_VERSION_FILE}")

# Find out the current version
IF ( HDK_VERSION_IN_SYS_SYS_VERSION_H )
  #
  FILE ( STRINGS "${HDK_VERSION_FILE}" hdk_major_version_str REGEX "^#define[\t ]+SYS_VERSION_MAJOR_INT[\t ]+.*")
  FILE ( STRINGS "${HDK_VERSION_FILE}" hdk_minor_version_str REGEX "^#define[\t ]+SYS_VERSION_MINOR_INT[\t ]+.*")
  FILE ( STRINGS "${HDK_VERSION_FILE}" hdk_build_version_str REGEX "^#define[\t ]+SYS_VERSION_BUILD_INT[\t ]+.*")
  FILE ( STRINGS "${HDK_VERSION_FILE}" hdk_patch_version_str REGEX "^#define[\t ]+SYS_VERSION_PATCH_INT[\t ]+.*")
  #
  STRING (REGEX REPLACE "^.*SYS_VERSION_MAJOR_INT[\t ]+([0-9]*).*$" "\\1"
	HDK_MAJOR_VERSION_STRING "${hdk_major_version_str}")
  STRING (REGEX REPLACE "^.*SYS_VERSION_MINOR_INT[\t ]+([0-9]*).*$" "\\1"
	HDK_MINOR_VERSION_STRING "${hdk_minor_version_str}")
  STRING (REGEX REPLACE "^.*SYS_VERSION_BUILD_INT[\t ]+([0-9]*).*$" "\\1"
	HDK_BUILD_VERSION_STRING "${hdk_build_version_str}")
  STRING (REGEX REPLACE "^.*SYS_VERSION_PATCH_INT[\t ]+([0-9]*).*$" "\\1"
	HDK_PATCH_VERSION_STRING "${hdk_patch_version_str}")
  #
  UNSET (hdk_major_version_str)
  UNSET (hdk_minor_version_str)
  UNSET (hdk_build_version_str)
  UNSET (hdk_patch_version_str)
ELSE ()
  #
  FILE ( STRINGS "${HDK_VERSION_FILE}" hdk_major_version_str REGEX "^#define[\t ]+UT_MAJOR_VERSION_INT[\t ]+.*")
  FILE ( STRINGS "${HDK_VERSION_FILE}" hdk_minor_version_str REGEX "^#define[\t ]+UT_MINOR_VERSION_INT[\t ]+.*")
  FILE ( STRINGS "${HDK_VERSION_FILE}" hdk_build_version_str REGEX "^#define[\t ]+UT_BUILD_VERSION_INT[\t ]+.*")
  FILE ( STRINGS "${HDK_VERSION_FILE}" hdk_patch_version_str REGEX "^#define[\t ]+UT_PATCH_VERSION_INT[\t ]+.*")
  #
  STRING (REGEX REPLACE "^.*UT_MAJOR_VERSION_INT[\t ]+([0-9]*).*$" "\\1"
	HDK_MAJOR_VERSION_STRING "${hdk_major_version_str}")
  STRING (REGEX REPLACE "^.*UT_MINOR_VERSION_INT[\t ]+([0-9]*).*$" "\\1"
	HDK_MINOR_VERSION_STRING "${hdk_minor_version_str}")
  STRING (REGEX REPLACE "^.*UT_BUILD_VERSION_INT[\t ]+([0-9]*).*$" "\\1"
	HDK_BUILD_VERSION_STRING "${hdk_build_version_str}")
  STRING (REGEX REPLACE "^.*UT_PATCH_VERSION_INT[\t ]+([0-9]*).*$" "\\1"
	HDK_PATCH_VERSION_STRING "${hdk_patch_version_str}")
  #
  UNSET (hdk_major_version_str)
  UNSET (hdk_minor_version_str)
  UNSET (hdk_build_version_str)
  UNSET (hdk_patch_version_str)
ENDIF ()

SET ( HDK_VERSION_MAJOR ${HDK_MAJOR_VERSION_STRING} CACHE STRING "HDK major version")
SET ( HDK_VERSION_MINOR ${HDK_MINOR_VERSION_STRING} CACHE STRING "HDK minor version")
SET ( HDK_VERSION_BUILD ${HDK_BUILD_VERSION_STRING} CACHE STRING "HDK build version")
SET ( HDK_VERSION "${HDK_MAJOR_VERSION_STRING}.${HDK_MINOR_VERSION_STRING}.${HDK_BUILD_VERSION_STRING}.${HDK_PATCH_VERSION_STRING}" CACHE STRING "HDK version")

# MESSAGE ( "HDK_VERSION = ${HDK_VERSION}")

FIND_PACKAGE_HANDLE_STANDARD_ARGS ( HDK
  REQUIRED_VARS HDK_LOCATION
  VERSION_VAR   HDK_VERSION
  )

IF (HDK_FOUND)

  IF (APPLE)
  ELSE ()
	SET ( HDK_HOME_HFS
	  $ENV{HOME}/houdini${HDK_MAJOR_VERSION_STRING}.${HDK_MINOR_VERSION_STRING}
	  )
  ENDIF()
  # MESSAGE ( "HDK_VERSION_STRING = ${HDK_VERSION_STRING}")
  SET ( HCUSTOM_COMMAND $ENV{HFS}/bin/hcustom ) 
  SET ( HOTL_COMMAND $ENV{HFS}/bin/hotl )

  SET ( HDK_INCLUDE_DIR "${HDK_LOCATION}/toolkit/include;${HDK_LOCATION}/toolkit/include/htools" CACHE STRING "HDK include directory" )

  IF ( HDK_VERSION VERSION_GREATER 14 )
	EXECUTE_PROCESS ( COMMAND ${HCUSTOM_COMMAND} -g -c OUTPUT_VARIABLE DEBUG_TEMP_DEFINITIONS )
	EXECUTE_PROCESS ( COMMAND ${HCUSTOM_COMMAND} -c OUTPUT_VARIABLE TEMP_DEFINITIONS )
	EXECUTE_PROCESS ( COMMAND ${HCUSTOM_COMMAND} -m OUTPUT_VARIABLE TEMP_LINK_FLAGS )
	STRING ( STRIP ${TEMP_LINK_FLAGS}  HDK_LINK_FLAGS )
  ELSE ()
	EXECUTE_PROCESS ( COMMAND ${HCUSTOM_COMMAND} -g OUTPUT_VARIABLE DEBUG_TEMP_DEFINITIONS )
  ENDIF ()

  IF (HDK_DEBUG_REGEX)
	# Keep this around, it is useful
	# MESSAGE ( "HDK_DEBUG_REGEX : START")
	# MESSAGE("TEMP_DEFINITIONS = ${TEMP_DEFINITIONS}")

	IF (WIN32)
	ELSE()
	  # Original : This handles strict x.y.z 3 component
      # Reference STRING ( REGEX REPLACE "-DVERSION=..[0-9]+.[0-9]+.[0-9]+.. " "" HDK_DEFINITIONS "${TEMP_DEFINITIONS}")
	  # Improves : This handles x.y.z 3 component and optionally w.x.y.z 4 components
      STRING ( REGEX REPLACE "-DVERSION=..[0-9]+.[0-9]+.[0-9]*.[0-9]+.. " "" HDK_DEFINITIONS "${TEMP_DEFINITIONS}")
	  # MESSAGE("HDK_DEFINITIONS = ${HDK_DEFINITIONS}")
	ENDIF()
	# MESSAGE ( "HDK_DEBUG_REGEX : END")
  ENDIF ()

  IF (WIN32)

    # Release flags
    # STRING ( REGEX REPLACE "-DVERSION=\"[0-9]+.[0-9]+.[0-9]+\" " "" HDK_DEFINITIONS_RAW "${TEMP_DEFINITIONS}")
    STRING ( REGEX REPLACE " -I \\.| -I \".*\"|-DVERSION=\"[0-9]+.[0-9]+.[0-9]+\" " "" HDK_DEFINITIONS_RAW "${TEMP_DEFINITIONS}")

    STRING ( STRIP ${HDK_DEFINITIONS_RAW} HDK_DEFINITIONS)
    # SET ( CMAKE_C_FLAGS ${HDK_DEFINITIONS})
    # SET ( CMAKE_CXX_FLAGS ${HDK_DEFINITIONS})

    # MESSAGE ( "HDK_DEFINITIONS = ${HDK_DEFINITIONS}" )
    # MESSAGE ( "CMAKE_C_FLAGS = ${CMAKE_C_FLAGS}")
    # MESSAGE ( "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

    # Debug flags
    STRING ( REGEX REPLACE "Making DEBUG version" "" STRIP_TEMP_DEFINITIONS "${DEBUG_TEMP_DEFINITIONS}")
    # STRING ( REGEX REPLACE "-DVERSION=\"[0-9]+.[0-9]+.[0-9]+\" " "" DEBUG_HDK_DEFINITIONS "${STRIP_TEMP_DEFINITIONS}")
    STRING ( REGEX REPLACE " -I \\.| -I \".*\"|-DVERSION=\"[0-9]+.[0-9]+.[0-9]+\" " "" DEBUG_HDK_DEFINITIONS "${STRIP_TEMP_DEFINITIONS}")
    # SET ( CMAKE_C_FLAGS_DEBUG ${DEBUG_HDK_DEFINITIONS})
    # SET ( CMAKE_CXX_FLAGS_DEBUG ${DEBUG_HDK_DEFINITIONS})

    # MESSAGE ( "DEBUG_TEMP_DEFINITIONS = ${DEBUG_TEMP_DEFINITIONS}")
    # MESSAGE ( "DEBUG_HDK_DEFINITIONS = ${DEBUG_HDK_DEFINITIONS}" )
    # MESSAGE ( "CMAKE_C_FLAGS_DEBUG = ${CMAKE_C_FLAGS_DEBUG}")
    # MESSAGE ( "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")

    ADD_DEFINITIONS ( ${HDK_DEFINITIONS} )

  ELSE (WIN32)

    # Release flags
    # STRING ( REGEX REPLACE "-DVERSION=..[0-9]+.[0-9]+.[0-9]+.. " "" HDK_DEFINITIONS "${TEMP_DEFINITIONS}")
    STRING ( REGEX REPLACE "-DVERSION=..[0-9]+.[0-9]+.[0-9]*.[0-9]+.. " "" HDK_DEFINITIONS "${TEMP_DEFINITIONS}")

    # Debug flags
    STRING ( REGEX REPLACE "Making debug version" "" STRIP_TEMP_DEFINITIONS "${DEBUG_TEMP_DEFINITIONS}")
    # STRING ( REGEX REPLACE "-DVERSION=..[0-9]+.[0-9]+.[0-9]+.. " "" DEBUG_HDK_DEFINITIONS "${STRIP_TEMP_DEFINITIONS}")
    STRING ( REGEX REPLACE "-DVERSION=..[0-9]+.[0-9]+.[0-9]*.[0-9]+.. " "" DEBUG_HDK_DEFINITIONS "${STRIP_TEMP_DEFINITIONS}")

	IF ( NOT HDK_VERSION_IN_SYS_SYS_VERSION_H )
	  # Very old HDK, need to set things manually
	  SET ( HDK_DEFINITIONS " -DDLLEXPORT= -D_GNU_SOURCE -DLINUX -DAMD64 -m64 -fPIC -DSIZEOF_VOID_P=8 -DSESI_LITTLE_ENDIAN -DENABLE_THREADS -DUSE_PTHREADS -D_REENTRANT -D_FILE_OFFSET_BITS=64 -c -DGCC4 -DGCC3 -Wno-deprecated")
	  SET ( DEBUG_HDK_DEFINITIONS " -DDLLEXPORT= -D_GNU_SOURCE -DLINUX -DAMD64 -m64 -fPIC -DSIZEOF_VOID_P=8 -DSESI_LITTLE_ENDIAN -DENABLE_THREADS -DUSE_PTHREADS -D_REENTRANT -D_FILE_OFFSET_BITS=64 -c -DGCC4 -DGCC3 -Wno-deprecated -g")
	ENDIF ()
    # MESSAGE ( "DEBUG_HDK_DEFINITIONS = ${DEBUG_HDK_DEFINITIONS}" )
	# MESSAGE ( "HDK_DEFINITIONS = ${HDK_DEFINITIONS} ")
    IF (CMAKE_BUILD_TYPE MATCHES Debug)
      ADD_DEFINITIONS ( "${DEBUG_HDK_DEFINITIONS}" )
    ELSE (CMAKE_BUILD_TYPE MATCHES Debug)
      ADD_DEFINITIONS ( "${HDK_DEFINITIONS}" )
    ENDIF (CMAKE_BUILD_TYPE MATCHES Debug)

  ENDIF (WIN32)

  IF (WIN32)
    FILE ( GLOB DSOLIB_A $ENV{HFS}/custom/houdini/dsolib/*.a )
    FILE ( GLOB DSOLIB_LIB $ENV{HFS}/custom/houdini/dsolib/*.lib )
	#ELSEIF (APPLE)
    # FILE ( GLOB DSOLIB_DYLIB $ENV{HFS}/../Libraries/*.dylib )
  ELSE (WIN32)
    # Linux/OSX 
    LINK_DIRECTORIES ( $ENV{HDSO} ) 
    # LINK_DIRECTORIES ( $ENV{HFS}/dsolib ) 
  ENDIF (WIN32)

  IF (APPLE)
    SET ( HDK_LIBRARY_TYPE SHARED )
  ELSE (APPLE)
    SET ( HDK_LIBRARY_TYPE SHARED )
  ENDIF (APPLE)

  FUNCTION ( HDK_CREATE_SESITAG
      # _input_name
      _src_name )
    SET ( _input_name abc )
    SET ( PYTHON_SCRIPT ${HDK_PACKAGE_DIR}/gen_sesitag.py )
    IF ( WIN32 )
      # Houdini uses the python bundled with the distribution, use that
      LIST(APPEND CMAKE_PROGRAM_PATH  "$ENV{HFS}/python27")
    ENDIF ()
    FIND_PROGRAM ( PYTHON_EXECUTABLE NAMES python )
    EXECUTE_PROCESS ( COMMAND "${PYTHON_EXECUTABLE}" ${PYTHON_SCRIPT} ${_src_name} )
    
  ENDFUNCTION ()

  FUNCTION ( HDK_ADD_EXECUTABLE _exe_NAME )

    IF (APPLE)
      SET ( HDK_LIBRARY_DIRS $ENV{HFS}/../Libraries )
      SET ( HDK_HOUDINI_LOCATION $ENV{HFS}/../Houdini )
    ELSE ()
      SET ( HDK_LIBRARY_DIRS $ENV{HFS}/dsolib )
      SET ( HDK_HOUDINI_LOCATION $ENV{HFS}/../Houdini )
    ENDIF ()
	IF ( HDK_AUTO_GENERATE_SESITAG )
      SET ( HDK_SESITAG_FILE ${CMAKE_BINARY_DIR}/${_exe_NAME}_sesitag.C )
      HDK_CREATE_SESITAG ( ${HDK_SESITAG_FILE} )
	ENDIF()
    ADD_EXECUTABLE ( ${_exe_NAME} ${ARGN} ${HDK_SESITAG_FILE} )
    IF (APPLE)
      SET_TARGET_PROPERTIES ( ${_exe_NAME} PROPERTIES
		LINK_FLAGS "${HDK_LINK_FLAGS} -L${HDK_LIBRARY_DIRS} ${HDK_HOUDINI_LOCATION}"
		)
    ELSEIF (WIN32)
      # windows
      TARGET_LINK_LIBRARIES ( ${_exe_NAME}
		${DSOLIB_A}
        ${DSOLIB_LIB}
        )
    ELSE()
      # Linux
      TARGET_LINK_LIBRARIES ( ${_exe_NAME}
		pthread
		HoudiniUI
		HoudiniOPZ
		HoudiniOP3
		HoudiniOP2
		HoudiniOP1
		HoudiniSIM
		HoudiniGEO
		HoudiniPRM
		HoudiniUT
		boost_system
		boost_program_options
		tbb
        )
    ENDIF (APPLE)
    
  ENDFUNCTION ()

  FUNCTION ( HDK_ADD_LIBRARY _lib_NAME )
    
    ADD_DEFINITIONS ( -DMAKING_DSO )

    SET ( HDK_LIBRARY_DIRS $ENV{HFS}/../Libraries )
    SET ( HDK_HOUDINI_LOCATION $ENV{HFS}/../Houdini )

    # MESSAGE ( "HDK_LIBRARY_DIRS = ${HDK_LIBRARY_DIRS}")
    # MESSAGE ( "HDK_HOUDINI_LOCATION = ${HDK_HOUDINI_LOCATION}")
	IF ( HDK_AUTO_GENERATE_SESITAG )
      SET ( HDK_SESITAG_FILE ${CMAKE_BINARY_DIR}/${_lib_NAME}_sesitag.C )
      HDK_CREATE_SESITAG ( ${HDK_SESITAG_FILE} )
	ENDIF ()
    ADD_LIBRARY ( ${_lib_NAME} ${HDK_LIBRARY_TYPE} ${ARGN} ${HDK_SESITAG_FILE} )
    IF (APPLE)
      SET_TARGET_PROPERTIES ( ${_lib_NAME} PROPERTIES
		LINK_FLAGS "${HDK_LINK_FLAGS} -L${HDK_LIBRARY_DIRS} ${HDK_HOUDINI_LOCATION}"
		PREFIX ""
		SUFFIX ".dylib"
		)
    ELSEIF (WIN32)
      # windows
      TARGET_LINK_LIBRARIES ( ${_lib_NAME}
		${DSOLIB_A}
        ${DSOLIB_LIB}
        )
    ELSE()
      # Linux
      SET_TARGET_PROPERTIES ( ${_lib_NAME} PROPERTIES
		PREFIX ""
		)
    ENDIF (APPLE)

  ENDFUNCTION ()

  FUNCTION ( HDK_ADD_STANDALONE_LIBRARY _lib_NAME )
    
    ADD_DEFINITIONS ( -DMAKING_DSO )

    SET ( HDK_LIBRARY_DIRS $ENV{HFS}/../Libraries )
    SET ( HDK_HOUDINI_LOCATION $ENV{HFS}/../Houdini )

    # MESSAGE ( "HDK_LIBRARY_DIRS = ${HDK_LIBRARY_DIRS}")
    # MESSAGE ( "HDK_HOUDINI_LOCATION = ${HDK_HOUDINI_LOCATION}")

    SET ( HDK_SESITAG_FILE ${CMAKE_BINARY_DIR}/${_lib_NAME}_sesitag.C )
    HDK_CREATE_SESITAG ( ${HDK_SESITAG_FILE} )

    ADD_LIBRARY ( ${_lib_NAME} ${HDK_LIBRARY_TYPE} ${ARGN} ${HDK_SESITAG_FILE} )
    IF (APPLE)
      SET_TARGET_PROPERTIES ( ${_lib_NAME} PROPERTIES
		LINK_FLAGS "${HDK_LINK_FLAGS} -L${HDK_LIBRARY_DIRS} ${HDK_HOUDINI_LOCATION}"
		PREFIX ""
		SUFFIX ".dylib"
		)
      TARGET_LINK_LIBRARIES ( ${_lib_NAME}
		HoudiniUI
		HoudiniOPZ
		HoudiniOP3
		HoudiniOP2
		HoudiniOP1
		HoudiniSIM
		HoudiniGEO
		HoudiniPRM
		HoudiniUT
		)
    ELSEIF (WIN32)
      # windows
      TARGET_LINK_LIBRARIES ( ${_lib_NAME}
		${DSOLIB_A}
        ${DSOLIB_LIB}
		)
      #TARGET_LINK_LIBRARIES ( ${_lib_NAME}
      #  ${DSOLIB_A}
      #  ${DSOLIB_LIB}
      #  )
    ELSE()
      # Linux
      SET_TARGET_PROPERTIES ( ${_lib_NAME} PROPERTIES
		PREFIX ""
		)
      TARGET_LINK_LIBRARIES ( ${_lib_NAME}
		HoudiniUI
		HoudiniOPZ
		HoudiniOP3
		HoudiniOP2
		HoudiniOP1
		HoudiniSIM
		HoudiniGEO
		HoudiniPRM
		HoudiniUT
		)
    ENDIF (APPLE)

  ENDFUNCTION ()

  FUNCTION ( HDK_MAKE_OTL _otl_NAME _op_NAME _label_NAME _ds_FILENAME _icn_FILENAME )
    SET ( DIR_NAME
      ${CMAKE_CURRENT_BINARY_DIR}/${_op_NAME}_otldir )
    FILE ( REMOVE_RECURSE ${DIR_NAME})
    FILE ( MAKE_DIRECTORY ${DIR_NAME})


	SET ( ARGS --directory ${DIR_NAME} --operator ${_op_NAME} --label ${_label_NAME} )
    IF ( _ds_FILENAME STRGREATER "" )
	  LIST ( APPEND ARGS --dialogscript ${_ds_FILENAME} )
	ENDIF ()
    IF ( _icn_FILENAME STRGREATER "" )
	  LIST ( APPEND ARGS --icon ${_icn_FILENAME} )
	ENDIF ()
	EXECUTE_PROCESS ( COMMAND ${CREATE_OTL_DIR_EXECUTABLE} ${ARGS} )
    EXECUTE_PROCESS ( COMMAND ${HOTL_COMMAND} -c ${DIR_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${_otl_NAME} )
  ENDFUNCTION ()

  FUNCTION ( HDK_COMBINE_OTLS _otl_NAME )
    FILE ( REMOVE ${_otl_NAME})
    FOREACH(arg ${ARGN})
      EXECUTE_PROCESS ( COMMAND ${HOTL_COMMAND} -M ${CMAKE_CURRENT_BINARY_DIR}/${arg} ${CMAKE_CURRENT_BINARY_DIR}/${_otl_NAME} )
	ENDFOREACH (arg)
  ENDFUNCTION ()

  # The vargs must be in the form "<operator-name>;<dso-name>" for each
  # of the pair to be successfully composed into the VRAYprocedural
  # file
  FUNCTION ( HDK_COMPOSE_VRAYPROCEDURAL_FILE _vrayprocedural_FILENAME )
	LIST ( LENGTH ARGN NUM_ITEMS )
	MATH ( EXPR NUM_ITEMS_MODULO "${NUM_ITEMS} % 2" )
	# MATH ( EXPR NUM_ITEMS_LESS_ONE "${NUM_ITEMS} - 1" )
	IF ( ${NUM_ITEMS_MODULO} EQUAL 0 )
	  MATH ( EXPR NUM_PAIRS "${NUM_ITEMS} / 2" )
	  MATH ( EXPR NUM_PAIRS_LESS_ONE "${NUM_PAIRS} - 1" )

	  FILE ( WRITE  ${_vrayprocedural_FILENAME} "// Procedural Insight Pty. Ltd.\n" )
	  FILE ( APPEND ${_vrayprocedural_FILENAME} "#if defined(WIN32)\n" )
	  FILE ( APPEND ${_vrayprocedural_FILENAME} "    #define DSO_FILE(filename)mantra/filename.dll\n" )
	  FILE ( APPEND ${_vrayprocedural_FILENAME} "#elif defined(MBSD)\n" )
	  FILE ( APPEND ${_vrayprocedural_FILENAME} "    #define DSO_FILE(filename)mantra/filename.dylib\n" )
	  FILE ( APPEND ${_vrayprocedural_FILENAME} "#else\n" )
	  FILE ( APPEND ${_vrayprocedural_FILENAME} "    #define DSO_FILE(filename)mantra/filename.so\n" )
	  FILE ( APPEND ${_vrayprocedural_FILENAME} "#endif\n" )

	  FOREACH ( PAIR_INDEX RANGE ${NUM_PAIRS_LESS_ONE} )
		MATH ( EXPR OPERATOR_ITEM_INDEX "${PAIR_INDEX} * 2" )
		MATH ( EXPR DSO_ITEM_INDEX "${OPERATOR_ITEM_INDEX} + 1" )
		LIST ( GET ARGN ${OPERATOR_ITEM_INDEX} OPERATOR_ITEM )
		LIST ( GET ARGN ${DSO_ITEM_INDEX} DSO_ITEM )
		FILE ( APPEND ${_vrayprocedural_FILENAME} "${OPERATOR_ITEM}\tDSO_FILE(${DSO_ITEM})\n")
	  ENDFOREACH ()

	  FILE ( APPEND ${_vrayprocedural_FILENAME} "#undef DSO_FILE\n\n" )
	  FILE ( APPEND ${_vrayprocedural_FILENAME} "#include \"$HFS/houdini/VRAYprocedural\"\n" )
	ENDIF ()
  ENDFUNCTION ()

ENDIF (HDK_FOUND)
