cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CUCO_BENCHS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

###################################################################################################
# Auto-detect available GPU compute architectures
set(GPU_ARCHS "" CACHE STRING "List of GPU architectures (semicolon-separated) to be compiled for. Empty string means to auto-detect the GPUs on the current system")

if("${GPU_ARCHS}" STREQUAL "")
  include(../cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()


###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    add_executable(${CMAKE_BENCH_NAME}
                   ${CMAKE_BENCH_SRC})
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${CMAKE_BENCH_NAME} PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
    target_link_libraries(${CMAKE_BENCH_NAME} benchmark benchmark_main pthread cuco)
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
endfunction(ConfigureBench)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}"
                    "${GBENCH_INCLUDE_DIR}")

link_directories("${GBENCH_LIBRARY_DIR}")

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