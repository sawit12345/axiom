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

cmake_minimum_required(VERSION 3.18)
project(axiomcuda_backend VERSION 0.1.0 LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(USE_CUDA "Enable CUDA support" ON)
option(CUDA_DEBUG "Enable CUDA debug info" OFF)

# Find packages
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

# CUDA configuration
if(USE_CUDA)
    enable_language(CUDA)
    
    # Set CUDA architecture flags
    # Support for sm_70 (Volta), sm_75 (Turing), sm_80, sm_86 (Ampere), sm_90 (Hopper)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
    
    # CUDA flags
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -O3
        --use_fast_math
        --compiler-options -fPIC
        -Xcompiler -Wall
        -Xcompiler -Wextra
    )
    
    if(CUDA_DEBUG)
        list(APPEND CUDA_NVCC_FLAGS -g -G)
    endif()
    
    # Set CUDA runtime library
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()

# C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Source files
set(CORE_SOURCES
    src/core/math.cpp
    src/core/tensor.cpp
    src/core/utils.cpp
)

set(CUDA_SOURCES
    src/cuda/kernels.cu
    src/cuda/math_kernels.cu
    src/cuda/linear_algebra.cu
    src/cuda/distribution_kernels.cu
)

set(VI_SOURCES
    src/vi/distribution.cpp
    src/vi/exponential/base.cpp
    src/vi/exponential/mvn.cpp
    src/vi/exponential/multinomial.cpp
    src/vi/exponential/delta.cpp
    src/vi/conjugate/base.cpp
    src/vi/conjugate/mvn.cpp
    src/vi/conjugate/multinomial.cpp
)

set(TRANSFORM_SOURCES
    src/transforms/transform.cpp
    src/transforms/linear_mng.cpp
)

set(MODEL_SOURCES
    src/models/smm.cpp
    src/models/rmm.cpp
    src/models/tmm.cpp
    src/models/imm.cpp
    src/models/mixture.cpp
    src/models/hybrid_mixture.cpp
)

# Create object libraries for each component
add_library(core_objects OBJECT ${CORE_SOURCES})
add_library(vi_objects OBJECT ${VI_SOURCES})
add_library(transform_objects OBJECT ${TRANSFORM_SOURCES})
add_library(model_objects OBJECT ${MODEL_SOURCES})

# CUDA object library
if(USE_CUDA)
    add_library(cuda_objects OBJECT ${CUDA_SOURCES})
    set_target_properties(cuda_objects PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Set properties for all object libraries
set(OBJECT_LIBRARIES core_objects vi_objects transform_objects model_objects)
if(USE_CUDA)
    list(APPEND OBJECT_LIBRARIES cuda_objects)
endif()

foreach(target ${OBJECT_LIBRARIES})
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
        ${CUDA_INCLUDE_DIRS}
    )
endforeach()

# Main library (static)
add_library(axiomcuda_static STATIC
    $<TARGET_OBJECTS:core_objects>
    $<TARGET_OBJECTS:vi_objects>
    $<TARGET_OBJECTS:transform_objects>
    $<TARGET_OBJECTS:model_objects>
)

if(USE_CUDA)
    target_sources(axiomcuda_static PRIVATE
        $<TARGET_OBJECTS:cuda_objects>
    )
endif()

set_target_properties(axiomcuda_static PROPERTIES
    OUTPUT_NAME axiomcuda
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(axiomcuda_static
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CUSOLVER_LIBRARIES}
    ${CUDA_CURAND_LIBRARIES}
)

# Main library (shared)
add_library(axiomcuda_shared SHARED
    $<TARGET_OBJECTS:core_objects>
    $<TARGET_OBJECTS:vi_objects>
    $<TARGET_OBJECTS:transform_objects>
    $<TARGET_OBJECTS:model_objects>
)

if(USE_CUDA)
    target_sources(axiomcuda_shared PRIVATE
        $<TARGET_OBJECTS:cuda_objects>
    )
endif()

set_target_properties(axiomcuda_shared PROPERTIES
    OUTPUT_NAME axiomcuda
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_link_libraries(axiomcuda_shared
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CUSOLVER_LIBRARIES}
    ${CUDA_CURAND_LIBRARIES}
)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(axiomcuda_backend
        bindings/bindings.cpp
        bindings/tensor_bindings.cpp
        bindings/distribution_bindings.cpp
        bindings/transform_bindings.cpp
        bindings/model_bindings.cpp
    )
    
    target_link_libraries(axiomcuda_backend PRIVATE
        axiomcuda_static
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CUSOLVER_LIBRARIES}
        ${CUDA_CURAND_LIBRARIES}
    )
    
    set_target_properties(axiomcuda_backend PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../axiomcuda
        CXX_VISIBILITY_PRESET hidden
    )
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS axiomcuda_static axiomcuda_shared
    EXPORT axiomcuda-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT axiomcuda-targets
    FILE axiomcuda-targets.cmake
    NAMESPACE axiomcuda::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/axiomcuda
)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Generate and install config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/axiomcuda-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/axiomcuda-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/axiomcuda
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/axiomcuda-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/axiomcuda-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/axiomcuda-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/axiomcuda
)

# Print configuration summary
message(STATUS "")
message(STATUS "Axiom CUDA Backend Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA enabled:   ${USE_CUDA}")
if(USE_CUDA)
    message(STATUS "  CUDA arch:      ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  CUDA version:   ${CUDA_VERSION}")
endif()
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Tests:          ${BUILD_TESTS}")
message(STATUS "")
