# find_package(doxygen REQUIRED dot  )
if(WIN32)
  set(DOXYGEN_EXECUTABLE "C:/Program Files/doxygen/bin/doxygen.exe")
else()
  set(DOXYGEN_EXECUTABLE "doxygen")
endif()

message(STATUS "DOXYGEN_EXECUTABLE = ${DOXYGEN_EXECUTABLE}")

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-config DOXYGEN_CONFIG_CONTENT)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config
     ${DOXYGEN_CONFIG_CONTENT})
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config
     "OUTPUT_DIRECTORY=\"${CMAKE_CURRENT_BINARY_DIR}/docs\"\n")

add_custom_target(
  nanovdb_docs
  SOURCES doxygen-config
          doc.md
          codingstyle.txt
          FAQ.md
          SourceTree.md
          HowToBuild.md
          HelloWorld.md
  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Generating API documentation with Doxygen"
  VERBATIM)

install(
  CODE "EXECUTE_PROCESS(COMMAND \"${CMAKE_COMMAND}\" --build  \"${PROJECT_BINARY_DIR}\" --target nanovdb_docs)"
)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/html DESTINATION docs)
