#=============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
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

###################################################################################################
### warpcore ######################################################################################
###################################################################################################

# Use rapids_cpm_find to find or clone warpcore
include("${rapids-cmake-dir}/cpm/find.cmake")
rapids_cpm_find(warpcore 0.0
    GLOBAL_TARGETS warpcore::warpcore
    CPM_ARGS
        GITHUB_REPOSITORY sleeepyjack/warpcore
        GIT_TAG master
        OPTIONS
            "WARPCORE_BUILD_TESTS OFF"
            "WARPCORE_BUILD_BENCHMARKS OFF"
            "WARPCORE_BUILD_EXAMPLES OFF"
)

###################################################################################################
function(ConfigureWarpcoreBench BENCH_NAME)
    add_executable(${BENCH_NAME} ${ARGN})
    set_target_properties(${BENCH_NAME} PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")
    target_include_directories(${BENCH_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}"
                                             "${CMAKE_CURRENT_SOURCE_DIR}/../..")
    target_compile_options(${BENCH_NAME} PRIVATE --compiler-options=-Wall --compiler-options=-Wextra
      --compiler-options=-Werror -Wno-deprecated-gpu-targets --expt-extended-lambda -lineinfo)
    # Add GCC-specific warning suppression only for GCC
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(${BENCH_NAME} PRIVATE -Xcompiler -Wno-subobject-linkage)
    endif()
    target_link_libraries(${BENCH_NAME} PRIVATE
                                        nvbench::main
                                        pthread
                                        cuco
                                        warpcore)
endfunction(ConfigureWarpcoreBench)

###################################################################################################
# - warpcore bloom_filter benchmarks --------------------------------------------------------------
ConfigureWarpcoreBench(WARPCORE_BLOOM_FILTER_BENCH
  warpcore_add_bench.cu
  warpcore_contains_bench.cu)

