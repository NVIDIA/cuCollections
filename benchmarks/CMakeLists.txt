#=============================================================================
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.5.2
  OPTIONS
    "BENCHMARK_ENABLE_TESTING Off"
    # The REGEX feature test fails when gbench's cmake is run under CPM w/ gcc5.4 because it doesn't assume C++11
    # Additionally, attempting to set the CMAKE_CXX_VERSION here doesn't propogate to the feature test build
    # Therefore, we just disable the feature test and assume platforms we care about have a regex impl available
    "RUN_HAVE_STD_REGEX 0" #
    "BENCHMARK_ENABLE_INSTALL OFF"
)

CPMAddPackage(
  NAME nvbench
  GITHUB_REPOSITORY NVIDIA/nvbench
  GIT_TAG main
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL YES
)

###################################################################################################
### compiler function #############################################################################
###################################################################################################

###################################################################################################
function(ConfigureBench BENCH_NAME BENCH_SRC)
    add_executable(${BENCH_NAME} "${BENCH_SRC}")
    set_target_properties(${BENCH_NAME} PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
    target_include_directories(${BENCH_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${BENCH_NAME} PRIVATE --compiler-options=-Wall --compiler-options=-Wextra
      --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -Wno-subobject-linkage)
    target_link_libraries(${BENCH_NAME} PRIVATE
                                        benchmark benchmark_main
                                        pthread
                                        cuco
                                        CUDA::cudart)
endfunction(ConfigureBench)

###################################################################################################
function(ConfigureNVBench BENCH_NAME)
    add_executable(${BENCH_NAME} ${ARGN})
    set_target_properties(${BENCH_NAME} PROPERTIES
                                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nvbenchmarks")
    target_include_directories(${BENCH_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${BENCH_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)
    target_link_libraries(${BENCH_NAME} PRIVATE
                                        nvbench::main
                                        pthread
                                        cuco)
endfunction(ConfigureNVBench)

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

###################################################################################################
# - dynamic_map benchmarks ------------------------------------------------------------------------
set(DYNAMIC_MAP_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/dynamic_map_bench.cu")
ConfigureBench(DYNAMIC_MAP_BENCH "${DYNAMIC_MAP_BENCH_SRC}")

###################################################################################################
# - static_map benchmarks -------------------------------------------------------------------------
ConfigureNVBench(STATIC_MAP_BENCH
  hash_table/static_map/contains_bench.cu
  hash_table/static_map/erase_bench.cu
  hash_table/static_map/find_bench.cu
  hash_table/static_map/insert_bench.cu)

###################################################################################################
# - static_multimap benchmarks --------------------------------------------------------------------
ConfigureNVBench(STATIC_MULTIMAP_BENCH
  hash_table/static_multimap/insert_bench.cu
  hash_table/static_multimap/retrieve_bench.cu
  hash_table/static_multimap/query_bench.cu
  hash_table/static_multimap/count_bench.cu)
