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

add_custom_target(cuco.all.headers)

function(cuco_add_header_test label definitions)
  set(config_prefix "cuco")
  
  file(GLOB_RECURSE headers
    RELATIVE "${CUCO_SOURCE_DIR}/include"
    CONFIGURE_DEPENDS
    "${CUCO_SOURCE_DIR}/include/cuco/*.cuh"
    "${CUCO_SOURCE_DIR}/include/cuco/*.hpp"
  )
  
  list(LENGTH headers headers_count)
  message(STATUS "Found ${headers_count} headers for testing: ${headers}")

  # List of headers that have known issues or are not meant to be included directly
  set(excluded_headers
    # Add any headers that should be excluded from testing here
    # Example: cuco/internal_header.cuh
  )
  
  # Remove excluded headers
  if(excluded_headers)
    list(REMOVE_ITEM headers ${excluded_headers})
  endif()

  set(headertest_target ${config_prefix}.headers.${label})

  # Generate header test sources
  set(header_srcs)
  foreach (header IN LISTS headers)
    set(header_src "${CMAKE_CURRENT_BINARY_DIR}/headers/${headertest_target}/${header}.cu")
    
    # Create the directory if it doesn't exist
    get_filename_component(header_dir "${header_src}" DIRECTORY)
    file(MAKE_DIRECTORY "${header_dir}")
    
    # Write simple test file that includes the header
    file(WRITE "${header_src}" "#include <${header}>\nint main() { return 0; }\n")
    list(APPEND header_srcs ${header_src})
  endforeach()

  # Create object library that compiles each header
  add_library(${headertest_target} OBJECT ${header_srcs})
  target_link_libraries(${headertest_target} PUBLIC cuco::cuco)
  if (definitions)
    target_compile_definitions(${headertest_target} PRIVATE ${definitions})
  endif()

  # Add required CUDA compiler flags for cuco
  target_compile_options(${headertest_target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  )

  # Macro collision checks are enabled to ensure cuco headers don't conflict with system headers

  add_dependencies(cuco.all.headers ${headertest_target})
endfunction()

# Base header test - ensure all headers compile cleanly
cuco_add_header_test(base "")
