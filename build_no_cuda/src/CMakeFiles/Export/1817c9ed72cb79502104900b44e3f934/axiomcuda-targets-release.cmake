#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "axiomcuda::axiomcuda_static" for configuration "Release"
set_property(TARGET axiomcuda::axiomcuda_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(axiomcuda::axiomcuda_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaxiomcuda.a"
  )

list(APPEND _cmake_import_check_targets axiomcuda::axiomcuda_static )
list(APPEND _cmake_import_check_files_for_axiomcuda::axiomcuda_static "${_IMPORT_PREFIX}/lib/libaxiomcuda.a" )

# Import target "axiomcuda::axiomcuda_shared" for configuration "Release"
set_property(TARGET axiomcuda::axiomcuda_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(axiomcuda::axiomcuda_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaxiomcuda.so.0.1.0"
  IMPORTED_SONAME_RELEASE "libaxiomcuda.so.0"
  )

list(APPEND _cmake_import_check_targets axiomcuda::axiomcuda_shared )
list(APPEND _cmake_import_check_files_for_axiomcuda::axiomcuda_shared "${_IMPORT_PREFIX}/lib/libaxiomcuda.so.0.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
