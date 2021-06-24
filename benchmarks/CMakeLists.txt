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

if (benchmark_ADDED)
  # patch google benchmark target
  set_target_properties(benchmark PROPERTIES CXX_STANDARD 14)
endif()

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
                                        cuco)
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
set(RBK_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/reduce_by_key/reduce_by_key.cu")
ConfigureBench(RBK_BENCH "${RBK_BENCH_SRC}")
