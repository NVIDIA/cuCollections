cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureExample EXAMPLE_NAME EXAMPLE_SRC)
    add_executable(${EXAMPLE_NAME} "${EXAMPLE_SRC}")
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
                                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
    target_include_directories(${EXAMPLE_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${EXAMPLE_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE cuco CUDA::cudart)
endfunction(ConfigureExample)

###################################################################################################
### Example sources ##################################################################################
###################################################################################################

ConfigureExample(STATIC_MAP_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/static_map_example.cu")
ConfigureExample(STATIC_MULTIMAP_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_multimap/static_multimap_example.cu")
