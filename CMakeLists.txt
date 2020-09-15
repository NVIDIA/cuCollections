#=============================================================================
# Copyright (c) 2018-2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cuCollections VERSION 0.0.1 LANGUAGES C CXX CUDA)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

###################################################################################################
# - find packages we depend on --------------------------------------------------------------------

find_package(CUDAToolkit REQUIRED)

include(cmake/Modules/CPM.cmake)

CPMFindPackage(
    NAME CUB
    GITHUB_REPOSITORY thrust/cub
    GIT_TAG main # TODO: Update to a tagged version of CUB that incorporates their cmake overhaul
    OPTIONS
      "CUB_ENABLE_HEADER_TESTING Off"
      "CUB_ENABLE_TESTING Off"
      "CUB_ENABLE_EXAMPLES Off"
      "CUB_DISABLE_ARCH_BY_DEFAULT On"
)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "${CMAKE_CXX_COMPILER}")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

###################################################################################################
# - cuco target   ---------------------------------------------------------------------------------
add_library(cuco INTERFACE)
target_include_directories(cuco INTERFACE 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(cuco INTERFACE CUDA::cudart CUB)


# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -Xcompiler -rdynamic")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

# Build options
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" ON)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)

include(CheckIncludeFiles)
include(CheckLibraryExists)

if(BUILD_TESTS)
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
endif(BUILD_TESTS)

###################################################################################################
# - add google benchmark --------------------------------------------------------------------------

if(BUILD_BENCHMARKS)
  include(ConfigureGoogleBenchmark)
  if(GBENCH_FOUND)
    message(STATUS "Google C++ Benchmarking Framework (Google Benchmark) found in ${GBENCH_ROOT}")
    include_directories(${GBENCH_INCLUDE_DIR})
    add_subdirectory(${CMAKE_SOURCE_DIR}/benchmarks)
  else()
    message(AUTHOR_WARNING "Google C++ Benchmarking Framework (Google Benchmark) not found: automated tests are disabled.")
  endif(GBENCH_FOUND)
endif(BUILD_BENCHMARKS)
