cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(CPM)


###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench BENCH_NAME BENCH_SRC)
    add_executable(${BENCH_NAME} "${BENCH_SRC}")
    set_target_properties(${BENCH_NAME} PROPERTIES 
                                        POSITION_INDEPENDENT_CODE ON
                                        CUDA_ARCHITECTURES ${GPU_ARCHS}
                                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/thirdparty_benchmarks")
    target_include_directories(${BENCH_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${BENCH_NAME} PRIVATE 
                                        benchmark benchmark_main 
                                        pthread)
endfunction(ConfigureBench)

###################################################################################################
### test sources ##################################################################################
###################################################################################################
