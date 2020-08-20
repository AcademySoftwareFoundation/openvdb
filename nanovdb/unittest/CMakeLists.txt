# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0

#---------------------------------------------------------------
# Unit tests
#

#-----------------------------------------------------------------------
# TODO: Benchmark should probably not require gtest.
if (NOT (NANOVDB_GTEST))
	message (WARNING " - GTest required to build unittests. Skipping.")
	return ()
endif()

# ---------------------------------------------------------------

SET (UNITTEST_NANOVDB_SOURCE_FILES "TestNanoVDB.cpp")

SET_SOURCE_FILES_PROPERTIES ( ${UNITTEST_NANOVDB_SOURCE_FILES}
  PROPERTIES
  COMPILE_FLAGS "-DNOMINMAX"
  )

ADD_EXECUTABLE ( testNanoVDB
	${UNITTEST_NANOVDB_SOURCE_FILES}
	)

TARGET_INCLUDE_DIRECTORIES ( testNanoVDB
	PUBLIC 
	)

TARGET_LINK_LIBRARIES ( testNanoVDB
	PRIVATE
	libnanovdb
	${NANOVDB_GTEST}
	)

ADD_TEST ( NanoVDB_unit_test testNanoVDB )

# ---------------------------------------------------------------

IF (NOT (TBB_FOUND AND OpenVDB_FOUND))
	MESSAGE(WARNING " - OpenVDB required to build OpenVDB unit tests. Skipping.")
ELSE()

	SET (UNITTEST_OPENVDB_SOURCE_FILES "TestOpenVDB.cpp")

	SET_SOURCE_FILES_PROPERTIES ( ${UNITTEST_OPENVDB_SOURCE_FILES}
	PROPERTIES
	COMPILE_FLAGS "-DNOMINMAX"
	)

	ADD_EXECUTABLE ( testOpenVDB
		${UNITTEST_OPENVDB_SOURCE_FILES}
		)

	TARGET_INCLUDE_DIRECTORIES ( testOpenVDB
		PUBLIC 
		)

	TARGET_LINK_LIBRARIES ( testOpenVDB    
		PRIVATE
		libnanovdb
		${NANOVDB_GTEST}
		${NANOVDB_TBB}
		${NANOVDB_OPENVDB}
		)

	ADD_TEST ( OpenVDB_unit_test testOpenVDB )
ENDIF()

# ---------------------------------------------------------------

# workaround for win32 bug when nvcc "--keep" is used.
IF (WIN32 AND NANOVDB_CUDA_KEEP_PTX)
	FILE (MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
ENDIF()

