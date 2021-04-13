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
    #GITHUB_REPOSITORY owensgroup/SlabAlloc
    GIT_TAG origin/resize
    #GIT_TAG origin/master
    DOWNLOAD_ONLY YES
)

add_library(SLABALLOC INTERFACE)
target_include_directories(SLABALLOC INTERFACE "${SLABALLOC_SOURCE_DIR}/src")

CPMAddPackage(
    NAME SLABHASH
    GITHUB_REPOSITORY Nicolas-Iskos/SlabHash
    GIT_TAG origin/resize
    DOWNLOAD_ONLY YES
)

add_library(SLABHASH INTERFACE)
target_include_directories(SLABHASH INTERFACE "${SLABHASH_SOURCE_DIR}/src")

CPMAddPackage(
    NAME WARPCORE
    GITHUB_REPOSITORY sleeepyjack/warpcore
    GIT_TAG origin/master
    DOWNLOAD_ONLY YES
)

add_library(WARPCORE INTERFACE)
target_include_directories(WARPCORE INTERFACE "${WARPCORE_SOURCE_DIR}/include")

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
                                        SLABHASH
                                        WARPCORE
                                        )

    target_compile_options(${BENCH_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)

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
set(DYNAMIC_H2H_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/dynamic_h2h_bench.cu")
ConfigureBench(DYNAMIC_H2H_MAP_BENCH "${DYNAMIC_H2H_BENCH_SRC}")

###################################################################################################
set(STATIC_H2H_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/static_h2h_bench.cu")
ConfigureBench(STATIC_H2H_MAP_BENCH "${STATIC_H2H_BENCH_SRC}")

###################################################################################################
set(SLABHASH_MAX_LF_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/slabhash_max_lf.cu")
ConfigureBench(SLABHASH_MAX_LF "${SLABHASH_MAX_LF_SRC}")

###################################################################################################