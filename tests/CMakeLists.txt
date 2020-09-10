cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUCO_TESTS LANGUAGES C CXX CUDA)


include(CTest)

###################################################################################################
# - Fetch Catch   ---------------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.11.1
)

FetchContent_GetProperties(catch)

if(NOT catch_POPULATED)
  FetchContent_Populate(catch)
  add_subdirectory(${catch_SOURCE_DIR} ${catch_BINARY_DIR})
endif()
set(CATCH_INCLUDE_DIR "${catch_SOURCE_DIR}/single_include" )

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CATCH_INCLUDE_DIR}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" 
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}")

# catch_main.cpp defines `CATCH_CONFIG_MAIN` which provides main() 
# Compiles it to be linked into test executables
add_library(CatchMain OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/catch_main.cpp ${CATCH_INCLUDE_DIR}/catch2/catch.hpp)


###################################################################################################
function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                   ${CMAKE_TEST_SRC}
                   $<TARGET_OBJECTS:CatchMain>
                   "${CATCH_INCLUDE_DIR}/catch2/catch.hpp")
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    #set(CU_COLLECTIONS_TEST_LIST ${CU_COLLECTIONS_TEST_LIST} ${CMAKE_TEST_NAME} CACHE INTERNAL "CU_COLLECTIONS_TEST_LIST")
endfunction(ConfigureTest)

###################################################################################################
### test sources ##################################################################################
###################################################################################################
set(STATIC_MAP_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/static_map/static_map_test.cu")

ConfigureTest(STATIC_MAP_TEST "${STATIC_MAP_TEST_SRC}")
####################################################################################################
set(DYNAMIC_MAP_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/dynamic_map/dynamic_map_test.cu")

ConfigureTest(DYNAMIC_MAP_TEST "${DYNAMIC_MAP_TEST_SRC}")
####################################################################################################