#=============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureExample EXAMPLE_NAME EXAMPLE_SRC)
    add_executable(${EXAMPLE_NAME} "${EXAMPLE_SRC}")
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
                                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
    target_include_directories(${EXAMPLE_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${EXAMPLE_NAME} PRIVATE --compiler-options=-Wall --compiler-options=-Wextra
      --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -Wno-subobject-linkage)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE cuco CUDA::cudart)
endfunction(ConfigureExample)

###################################################################################################
### Example sources ###############################################################################
###################################################################################################

ConfigureExample(HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/host_bulk_example.cu")
ConfigureExample(DEVICE_SIDE_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/device_view_example.cu")
ConfigureExample(CUSTOM_TYPE_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/custom_type_example.cu")
ConfigureExample(STATIC_MULTIMAP_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_multimap/static_multimap_example.cu")
ConfigureExample(BLOOM_FILTER_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/bloom_filter/bloom_filter_example.cu")

foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if("${arch}" MATCHES "^6")
      target_compile_definitions(CUSTOM_TYPE_EXAMPLE PRIVATE CUCO_NO_INDEPENDENT_THREADS)
      break()
    endif()
endforeach()
