cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

###################################################################################################
# Auto-detect available GPU compute architectures
set(GPU_ARCHS "" CACHE STRING "List of GPU architectures (semicolon-separated) to be compiled for. Empty string means to auto-detect the GPUs on the current system")

if("${GPU_ARCHS}" STREQUAL "")
  include(../cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureExample EXAMPLE_NAME EXAMPLE_SRC)
    add_executable(${EXAMPLE_NAME} "${EXAMPLE_SRC}")
    set_target_properties(${EXAMPLE_NAME} PROPERTIES 
                                          CUDA_ARCHITECTURES ${GPU_ARCHS}
                                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
    target_include_directories(${EXAMPLE_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${EXAMPLE_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE cuco)
endfunction(ConfigureExample)

###################################################################################################
### Example sources ##################################################################################
###################################################################################################

ConfigureExample(STATIC_MAP_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/static_map_example.cu")
ConfigureExample(STATIC_MULTIMAP_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_multimap/static_multimap_example.cu")
