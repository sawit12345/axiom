# Copyright 2025 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the "License");
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/axiom/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Root CMakeLists.txt for axiomcuda package
# This is a wrapper that includes the main build configuration from src/

cmake_minimum_required(VERSION 3.18)
project(axiomcuda VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(USE_CUDA "Enable CUDA support" ON)
option(CUDA_DEBUG "Enable CUDA debug info" OFF)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 (usually installed via pip for Python builds)
find_package(pybind11 REQUIRED)

# CUDA detection and configuration
if(USE_CUDA)
    # Check for CUDA
    find_package(CUDA QUIET)
    
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
        message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
        
        # Enable CUDA language
        enable_language(CUDA)
        
        # CUDA Standard
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        
        # Set CUDA architecture flags
        # Support for sm_70 (Volta), sm_75 (Turing), sm_80, sm_86 (Ampere), sm_90 (Hopper)
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
        endif()
        
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        
        # Set CUDA runtime library
        set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    else()
        message(WARNING "CUDA not found. Building with CPU-only support.")
        set(USE_CUDA OFF)
    endif()
else()
    message(STATUS "CUDA support disabled (USE_CUDA=OFF)")
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "AXIOMCUDA Build Configuration:")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA enabled:      ${USE_CUDA}")
if(USE_CUDA AND CUDA_FOUND)
    message(STATUS "  CUDA version:      ${CUDA_VERSION}")
    message(STATUS "  CUDA arch:         ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Python:            ${Python_VERSION}")
message(STATUS "  Python executable: ${Python_EXECUTABLE}")
message(STATUS "  Python bindings:   ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Tests:             ${BUILD_TESTS}")
message(STATUS "")

# Add the src/ subdirectory which contains the main build configuration
add_subdirectory(src)

# Installation rules for the Python package
if(BUILD_PYTHON_BINDINGS)
    # The Python module will be built in src/ and output to axiomcuda/
    # We need to ensure it's properly installed
    
    # Get the extension suffix from Python
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
        OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Install the built module to the correct location
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/axiomcuda/axiomcuda_backend${PYTHON_EXT_SUFFIX}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/axiomcuda/
        OPTIONAL
    )
endif()

# Install headers (for C++ library usage)
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    OPTIONAL
)

# Install the CMake config file for find_package support
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/axiomcuda-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/axiomcuda-config.cmake
    INSTALL_DESTINATION lib/cmake/axiomcuda
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/axiomcuda-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/axiomcuda-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/axiomcuda-config-version.cmake
    DESTINATION lib/cmake/axiomcuda
    OPTIONAL
)
