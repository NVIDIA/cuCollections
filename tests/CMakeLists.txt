#=============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

include(CTest)

###################################################################################################
# - Fetch Catch   ---------------------------------------------------------------------------------

CPMAddPackage(
  NAME Catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 2.11.1
)

if(Catch2_ADDED)
    include(${Catch2_SOURCE_DIR}/contrib/Catch.cmake)
endif()

# catch_main.cpp defines `CATCH_CONFIG_MAIN` which provides main()
# Compiles it to be linked into test executables
add_library(CatchMain OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/catch_main.cpp)
target_link_libraries(CatchMain Catch2::Catch2)

###################################################################################################
function(ConfigureTest TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN}
                   $<TARGET_OBJECTS:CatchMain>) # Link in the CatchMain object file
    target_link_libraries(${TEST_NAME} Catch2::Catch2 cuco CUDA::cudart)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${TEST_NAME} PROPERTIES
                                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    target_compile_options(${TEST_NAME} PRIVATE --compiler-options=-Wall --compiler-options=-Wextra
      --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -Wno-subobject-linkage)
    catch_discover_tests(${TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - static_map tests ------------------------------------------------------------------------------
ConfigureTest(STATIC_MAP_TEST
    static_map/custom_type_test.cu
    static_map/duplicate_keys_test.cu
    static_map/erase_test.cu
    static_map/heterogeneous_lookup_test.cu
    static_map/key_sentinel_test.cu
    static_map/shared_memory_test.cu
    static_map/stream_test.cu
    static_map/unique_sequence_test.cu)

foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if("${arch}" MATCHES "^6")
      target_compile_definitions(STATIC_MAP_TEST PRIVATE CUCO_NO_INDEPENDENT_THREADS)
      break()
    endif()
endforeach()

###################################################################################################
# - dynamic_map tests -----------------------------------------------------------------------------
ConfigureTest(DYNAMIC_MAP_TEST
    dynamic_map/unique_sequence_test.cu)

###################################################################################################
# - static_multimap tests -------------------------------------------------------------------------
ConfigureTest(STATIC_MULTIMAP_TEST
    static_multimap/custom_pair_retrieve_test.cu
    static_multimap/custom_type_test.cu
    static_multimap/heterogeneous_lookup_test.cu
    static_multimap/insert_if_test.cu
    static_multimap/multiplicity_test.cu
    static_multimap/non_match_test.cu
    static_multimap/pair_function_test.cu)
