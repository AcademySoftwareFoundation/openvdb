# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Resolve paths from this module so standalone NanoVDB builds work too.
function(openvdb_install_licenses destination)
  set(source_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
  install(FILES
    "${source_root}/LICENSE"
    "${source_root}/ext/THIRD-PARTY.md"
    DESTINATION "${destination}"
  )
  install(FILES "${source_root}/ext/blosc/openvdb/ext/blosc/LICENSE.txt"
    DESTINATION "${destination}/licenses/blosc")
  install(FILES "${source_root}/ext/blosc/openvdb/ext/blosc/lz4/LICENSE"
    DESTINATION "${destination}/licenses/lz4")
  install(FILES "${source_root}/ext/blosc/openvdb/ext/blosc/LICENSES/FASTLZ.txt"
    DESTINATION "${destination}/licenses/fastlz")
endfunction()
