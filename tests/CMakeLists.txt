cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(CTest)

###################################################################################################
# - Fetch Catch   ---------------------------------------------------------------------------------

CPMAddPackage(
  NAME Catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 2.11.1
)

if(Catch2_ADDED)
    include(${Catch2_SOURCE_DIR}/contrib/Catch.cmake)
endif()

# catch_main.cpp defines `CATCH_CONFIG_MAIN` which provides main()
# Compiles it to be linked into test executables
add_library(CatchMain OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/catch_main.cpp)
target_link_libraries(CatchMain Catch2::Catch2)

###################################################################################################
function(ConfigureTest TEST_NAME TEST_SRC)
    add_executable(${TEST_NAME}
                   "${TEST_SRC}"
                   $<TARGET_OBJECTS:CatchMain>) # Link in the CatchMain object file
    target_link_libraries(${TEST_NAME} Catch2::Catch2 cuco)
    set_target_properties(${TEST_NAME} PROPERTIES
                                       CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
                                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    target_compile_options(${TEST_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -Wno-subobject-linkage)
    catch_discover_tests(${TEST_NAME})
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
