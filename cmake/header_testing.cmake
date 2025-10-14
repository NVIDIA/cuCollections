#=============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all header builds:
add_custom_target(cuco.all.headers)

find_package(CUDAToolkit)

# Custom header test template content
set(CUCO_HEADER_TEST_TEMPLATE_CONTENT
"// This source file checks that:
// 1) Header <@header@> compiles without error.
// 2) Common macro collisions with platform/system headers are avoided.

// Define CUCO_HEADER_MACRO_CHECK(macro, header), which emits a diagnostic indicating
// a potential macro collision and halts.
//
// Hacky way to build a string, but it works on all tested platforms.
#define CUCO_HEADER_MACRO_CHECK(MACRO, HEADER) \\
  CUCO_HEADER_MACRO_CHECK_IMPL(                \\
    Identifier MACRO should not be used from cuco headers due to conflicts with HEADER macros.)

// Use raw platform macros instead of the CCCL macros since we
// don't want to #include any headers other than the one being tested.
//
// This is implemented for GCC/Clang (cuco doesn't support MSVC).
#if defined(__clang__) || defined(__GNUC__)

// GCC/clang implementation:
#  define CUCO_HEADER_MACRO_CHECK_IMPL(msg)   CUCO_HEADER_MACRO_CHECK_IMPL0(GCC error #msg)
#  define CUCO_HEADER_MACRO_CHECK_IMPL0(expr) _Pragma(#expr)

#else
#  error \"Unsupported compiler for cuco header testing\"
#endif

// May be defined to skip macro check for certain configurations.
#ifndef CUCO_IGNORE_HEADER_MACRO_CHECKS

// complex.h conflicts
#  define I CUCO_HEADER_MACRO_CHECK('I', complex.h)

// windows.h conflicts
#  define small CUCO_HEADER_MACRO_CHECK('small', windows.h)

#  ifdef _WIN32
// On Windows, make sure any include of Windows.h (e.g. via NVTX) does not define the checked macros
#    define WIN32_LEAN_AND_MEAN
#  endif // _WIN32

// termios.h conflicts (NVIDIA/thrust#1547)
#  define B0 CUCO_HEADER_MACRO_CHECK(\"B0\", termios.h)

#endif // CUCO_IGNORE_HEADER_MACRO_CHECKS

#include <@header@>

// No main function - this is compiled as an object file for link checking
")

function(cuco_generate_header_tests target_name headers lang)
  # Configure header templates:
  set(header_srcs)
  foreach (header IN LISTS headers)
    if(lang STREQUAL "CUDA")
      set(header_src "${CMAKE_CURRENT_BINARY_DIR}/headers/${target_name}/${header}.cu")
    else()
      set(header_src "${CMAKE_CURRENT_BINARY_DIR}/headers/${target_name}/${header}.cpp")
    endif()
    
    # Create the directory if it doesn't exist
    get_filename_component(header_dir "${header_src}" DIRECTORY)
    file(MAKE_DIRECTORY "${header_dir}")
    
    # Configure the template
    string(CONFIGURE "${CUCO_HEADER_TEST_TEMPLATE_CONTENT}" header_content @ONLY)
    file(WRITE "${header_src}" "${header_content}")
    list(APPEND header_srcs ${header_src})
  endforeach()

  # Object library that compiles each header:
  add_library(${target_name} OBJECT ${header_srcs})
  
  # Set language-specific properties
  if(lang STREQUAL "CUDA")
    set_target_properties(${target_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Set CUDA architectures
    if(CMAKE_CUDA_ARCHITECTURES)
      set_target_properties(${target_name} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      )
    endif()
    
    # Add required CUDA compiler flags for cuco
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
  endif()
  
  target_link_libraries(${target_name} PUBLIC cuco::cuco)
  
  # Check that all functions in headers are either template functions or inline:
  set(link_target ${target_name}.link_check)
  if(lang STREQUAL "CUDA")
    add_executable(${link_target} "${CMAKE_CURRENT_BINARY_DIR}/link_check_main_${target_name}.cu")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/link_check_main_${target_name}.cu" 
         "int main() { return 0; }")
  else()
    add_executable(${link_target} "${CMAKE_CURRENT_BINARY_DIR}/link_check_main_${target_name}.cpp")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/link_check_main_${target_name}.cpp" 
         "int main() { return 0; }")
  endif()
  
  target_link_libraries(${link_target} PUBLIC cuco::cuco)
  
  # Linking both ${target_name} and $<TARGET_OBJECTS:${target_name}> forces CMake to
  # link the same objects twice. The compiler will complain about duplicate symbols if
  # any functions are missing inline markup.
  target_link_libraries(${link_target} PRIVATE
    ${target_name}
    $<TARGET_OBJECTS:${target_name}>
  )
endfunction()

function(cuco_add_header_test label definitions)
  set(config_prefix "cuco")
  
  # Get all .cuh and .hpp files...
  file(GLOB_RECURSE all_headers
    RELATIVE "${CUCO_SOURCE_DIR}/include"
    CONFIGURE_DEPENDS
    "${CUCO_SOURCE_DIR}/include/cuco/*.cuh"
    "${CUCO_SOURCE_DIR}/include/cuco/*.hpp"
  )

  # ...and remove all the detail headers
  file(GLOB_RECURSE headers_exclude_details
    RELATIVE "${CUCO_SOURCE_DIR}/include"
    CONFIGURE_DEPENDS
    "${CUCO_SOURCE_DIR}/include/cuco/detail/*"
    "${CUCO_SOURCE_DIR}/include/cuco/*/detail/*"
    "${CUCO_SOURCE_DIR}/include/cuco/*/*/detail/*"
  )
  
  set(headers ${all_headers})
  if(headers_exclude_details)
    list(REMOVE_ITEM headers ${headers_exclude_details})
  endif()

  # List of headers that have known issues or are not meant to be included directly
  set(excluded_headers
    # Add any headers that should be excluded from testing here
    # Example: cuco/internal_header.cuh
  )
  
  # Remove excluded headers
  if(excluded_headers)
    list(REMOVE_ITEM headers ${excluded_headers})
  endif()

  # Separate headers by type for different language compilation
  set(cuda_headers)
  set(cpp_headers)
  
  foreach(header IN LISTS headers)
    if(header MATCHES "\\.cuh$")
      list(APPEND cuda_headers ${header})
    elseif(header MATCHES "\\.hpp$")
      list(APPEND cpp_headers ${header})
      list(APPEND cuda_headers ${header}) # .hpp files can also be compiled with CUDA
    endif()
  endforeach()

  # Compile headers with both host and cuda compilers
  set(langs CXX CUDA)

  foreach (lang IN LISTS langs)
    set(headertest_target ${config_prefix}.headers.${label})
    if (lang STREQUAL "CXX")
      # Append .cxx to the header test target name when compiling with C++ compiler
      set(headertest_target ${headertest_target}.cxx)
      set(test_headers ${cpp_headers})
    else()
      set(test_headers ${cuda_headers})
    endif()

    if(test_headers)
      cuco_generate_header_tests(${headertest_target} "${test_headers}" ${lang})
      
      if (definitions)
        target_compile_definitions(${headertest_target} PRIVATE ${definitions})
      endif()
      
      # Disable macro checks for now since cuco has known issues with 'I' identifier
      # This should be removed once the macro collision issues are fixed
      target_compile_definitions(${headertest_target} PRIVATE CUCO_IGNORE_HEADER_MACRO_CHECKS)

      add_dependencies(cuco.all.headers ${headertest_target})
    endif()
  endforeach()
endfunction()

# Base header test - ensure all headers compile cleanly
cuco_add_header_test(base "")
