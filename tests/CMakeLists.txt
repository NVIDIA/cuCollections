cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(cuDataStructures_TESTS LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()


set(cuDataStructures_TEST_LIST CACHE INTERNAL "cuDataStructures_TEST_LIST")

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                   ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread cuds)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    set(cuDataStructures_TEST_LIST ${cuDataStructures_TEST_LIST} ${CMAKE_TEST_NAME} CACHE INTERNAL "cuDataStructures_TEST_LIST")
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/libcudacxx/include"
                    "${GTEST_INCLUDE_DIR}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GTEST_LIBRARY_DIR}")

###################################################################################################
### test sources ##################################################################################
###################################################################################################
set(HASH_MAP_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/hash_map/map_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/hash_map/multimap_test.cu")

ConfigureTest(HASH_MAP_TEST "${HASH_MAP_TEST_SRC}")