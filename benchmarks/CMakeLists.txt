cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

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

#if (benchmark_ADDED)
#  # patch google benchmark target
#  set_target_properties(benchmark PROPERTIES CXX_STANDARD 14)
#endif()

CPMAddPackage(
  NAME nvbench
  GITHUB_REPOSITORY NVIDIA/nvbench
  GIT_TAG main
  GIT_SHALLOW TRUE
)

set_target_properties(benchmark PROPERTIES CXX_STANDARD 17)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench BENCH_NAME BENCH_SRC)
    add_executable(${BENCH_NAME} "${BENCH_SRC}")
    set_target_properties(${BENCH_NAME} PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
    target_include_directories(${BENCH_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${BENCH_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -Wno-subobject-linkage)
    target_link_libraries(${BENCH_NAME} PRIVATE
                                        benchmark benchmark_main
                                        pthread
                                        cuco
                                        CUDA::cudart)
endfunction(ConfigureBench)

###################################################################################################
function(ConfigureNVBench BENCH_NAME BENCH_SRC)
    add_executable(${BENCH_NAME} "${BENCH_SRC}")
    set_target_properties(${BENCH_NAME} PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nvbenchmarks")
    target_include_directories(${BENCH_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
                                             #"${NVBench_SOURCE_DIR}")
    target_compile_options(${BENCH_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)
    target_link_libraries(${BENCH_NAME} PRIVATE
                                        nvbench::main
                                        pthread
                                        cuco)
endfunction(ConfigureNVBench)

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
set(STATIC_REDUCTION_MAP_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/static_reduction_map_bench.cu")
ConfigureNVBench(STATIC_REDUCTION_MAP_BENCH "${STATIC_REDUCTION_MAP_BENCH_SRC}")

###################################################################################################
set(CUCO_RBK_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/reduce_by_key/cuco_reduce_by_key_bench.cu")
ConfigureNVBench(CUCO_RBK_BENCH "${CUCO_RBK_BENCH_SRC}")

###################################################################################################
set(THRUST_RBK_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/reduce_by_key/thrust_reduce_by_key_bench.cu")
ConfigureNVBench(THRUST_RBK_BENCH "${THRUST_RBK_BENCH_SRC}")

###################################################################################################
set(CUB_RBK_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/reduce_by_key/cub_reduce_by_key_bench.cu")
ConfigureNVBench(CUB_RBK_BENCH "${CUB_RBK_BENCH_SRC}")

###################################################################################################
set(STATIC_REDUCTION_MAP_PARAM_GRID_SEARCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/static_reduction_map_param_grid_search.cu")
ConfigureNVBench(STATIC_REDUCTION_MAP_PARAM_GRID_SEARCH "${STATIC_REDUCTION_MAP_PARAM_GRID_SEARCH_SRC}")