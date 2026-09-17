#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureExample EXAMPLE_NAME EXAMPLE_SRC)
    add_executable(${EXAMPLE_NAME} "${EXAMPLE_SRC}")
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
                                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
    target_include_directories(${EXAMPLE_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    cuco_set_common_compile_options(${EXAMPLE_NAME})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE cuco CUDA::cudart)
endfunction(ConfigureExample)

###################################################################################################
### Example sources ###############################################################################
###################################################################################################

ConfigureExample(STATIC_SET_HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_set/host_bulk_example.cu")
ConfigureExample(STATIC_SET_DEVICE_REF_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_set/device_ref_example.cu")
ConfigureExample(STATIC_SET_DEVICE_SUBSETS_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_set/device_subsets_example.cu")
ConfigureExample(STATIC_SET_SHARED_MEMORY_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_set/shared_memory_example.cu")
ConfigureExample(STATIC_SET_MAPPING_TABLE_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_set/mapping_table_example.cu")
ConfigureExample(STATIC_MULTISET_HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_multiset/host_bulk_example.cu")
ConfigureExample(STATIC_MAP_HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/host_bulk_example.cu")
ConfigureExample(STATIC_MAP_DEVICE_SIDE_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/device_ref_example.cu")
ConfigureExample(STATIC_MAP_CUSTOM_TYPE_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/custom_type_example.cu")
ConfigureExample(STATIC_MAP_COUNT_BY_KEY_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/count_by_key_example.cu")
ConfigureExample(STATIC_MAP_PREALLOCATED_MEMORY_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/preallocated_memory_example.cu")
ConfigureExample(STATIC_MAP_HETEROGENEOUS_LOOKUP_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/heterogeneous_lookup_example.cu")
ConfigureExample(STATIC_MULTIMAP_HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_multimap/host_bulk_example.cu")
ConfigureExample(HYPERLOGLOG_HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/hyperloglog/host_bulk_example.cu")
ConfigureExample(HYPERLOGLOG_DEVICE_REF_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/hyperloglog/device_ref_example.cu")
ConfigureExample(BLOOM_FILTER_HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/bloom_filter/host_bulk_example.cu")
ConfigureExample(BLOOM_FILTER_PERSISTING_L2_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/bloom_filter/persisting_l2_example.cu")
ConfigureExample(ROARING_BITMAP_HOST_BULK_FROM_INDICES_EXAMPLE
                 "${CMAKE_CURRENT_SOURCE_DIR}/roaring_bitmap/host_bulk_from_indices_example.cu")
ConfigureExample(ROARING_BITMAP_HOST_BULK_FROM_SERIALIZED_EXAMPLE
                 "${CMAKE_CURRENT_SOURCE_DIR}/roaring_bitmap/host_bulk_from_serialized_example.cu")
