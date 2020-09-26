cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(CPM)

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
)

if (benchmark_ADDED)
  # patch google benchmark target
  set_target_properties(benchmark PROPERTIES CXX_STANDARD 14)
endif()


CPMAddPackage(
    NAME SLABALLOC
    GITHUB_REPOSITORY Nicolas-Iskos/SlabAlloc
    GIT_TAG origin/grow_superblocks
    DOWNLOAD_ONLY YES
)

add_library(SLABALLOC INTERFACE)
target_include_directories(SLABALLOC INTERFACE "${SLABALLOC_SOURCE_DIR}/src")

CPMAddPackage(
    NAME SLABHASH
    GITHUB_REPOSITORY Nicolas-Iskos/SlabHash
    GIT_TAG origin/grow_superblocks
    DOWNLOAD_ONLY YES
)

add_library(SLABHASH INTERFACE)
target_include_directories(SLABHASH INTERFACE "${SLABHASH_SOURCE_DIR}/src")

###################################################################################################
# Auto-detect available GPU compute architectures
set(GPU_ARCHS "" CACHE STRING "List of GPU architectures (semicolon-separated) to be compiled for. Empty string means to auto-detect the GPUs on the current system")

if("${GPU_ARCHS}" STREQUAL "")
  include(../cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench BENCH_NAME BENCH_SRC)
    add_executable(${BENCH_NAME} "${BENCH_SRC}")
    set_target_properties(${BENCH_NAME} PROPERTIES 
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_ARCHITECTURES ${GPU_ARCHS}
                                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
    target_include_directories(${BENCH_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${BENCH_NAME} PRIVATE 
                                        benchmark benchmark_main 
                                        pthread 
                                        cuco
                                        SLABALLOC
                                        SLABHASH)
endfunction(ConfigureBench)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
set(DYNAMIC_MAP_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/dynamic_map_bench.cu")
ConfigureBench(DYNAMIC_MAP_BENCH "${DYNAMIC_MAP_BENCH_SRC}")

###################################################################################################
set(STATIC_MAP_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/static_map_bench.cu")
ConfigureBench(STATIC_MAP_BENCH "${STATIC_MAP_BENCH_SRC}")

###################################################################################################
set(LEGACY_STATIC_MAP_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/legacy_static_map_bench.cu")
ConfigureBench(LEGACY_STATIC_MAP_BENCH "${LEGACY_STATIC_MAP_BENCH_SRC}")

###################################################################################################
set(HEAD_TO_HEAD_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/head_to_head_bench.cu")
ConfigureBench(HEAD_TO_HEAD_MAP_BENCH "${HEAD_TO_HEAD_BENCH_SRC}")

###################################################################################################