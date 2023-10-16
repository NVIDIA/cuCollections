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
function(ConfigureBench BENCH_NAME)
    add_executable(${BENCH_NAME} ${ARGN})
    set_target_properties(${BENCH_NAME} PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")
    target_include_directories(${BENCH_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${BENCH_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr -lineinfo)
    target_link_libraries(${BENCH_NAME} PRIVATE
                                        nvbench::main
                                        pthread
                                        cuco)
endfunction(ConfigureBench)

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

###################################################################################################
# - static_set benchmarks -------------------------------------------------------------------------
ConfigureBench(STATIC_SET_BENCH
  hash_table/static_set/contains_bench.cu
  hash_table/static_set/find_bench.cu
  hash_table/static_set/insert_bench.cu
  hash_table/static_set/retrieve_all_bench.cu
  hash_table/static_set/size_bench.cu
  hash_table/static_set/rehash_bench.cu)

###################################################################################################
# - static_map benchmarks -------------------------------------------------------------------------
ConfigureBench(STATIC_MAP_BENCH
  hash_table/static_map/insert_bench.cu
  hash_table/static_map/find_bench.cu
  hash_table/static_map/contains_bench.cu
  hash_table/static_map/erase_bench.cu
  hash_table/static_map/insert_or_apply_bench.cu)

###################################################################################################
# - static_multimap benchmarks --------------------------------------------------------------------
ConfigureBench(STATIC_MULTIMAP_BENCH
  hash_table/static_multimap/insert_bench.cu
  hash_table/static_multimap/retrieve_bench.cu
  hash_table/static_multimap/query_bench.cu
  hash_table/static_multimap/count_bench.cu)

###################################################################################################
# - dynamic_map benchmarks ------------------------------------------------------------------------
ConfigureBench(DYNAMIC_MAP_BENCH
  hash_table/dynamic_map/insert_bench.cu
  hash_table/dynamic_map/find_bench.cu
  hash_table/dynamic_map/contains_bench.cu
  hash_table/dynamic_map/erase_bench.cu)

###################################################################################################
# - hash function benchmarks ----------------------------------------------------------------------
ConfigureBench(HASH_BENCH
  hash_bench.cu)
