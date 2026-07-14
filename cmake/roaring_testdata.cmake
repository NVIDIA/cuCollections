# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Only act if enabled
if(NOT CUCO_DOWNLOAD_ROARING_TESTDATA)
  return()
endif()

set(CUCO_ROARING_DATA_DIR "${CMAKE_BINARY_DIR}/data/roaring_bitmap")

file(MAKE_DIRECTORY "${CUCO_ROARING_DATA_DIR}")

set(ROARING_FORMATSPEC_BASE "https://raw.githubusercontent.com/RoaringBitmap/RoaringFormatSpec/5177ad9")

rapids_cmake_download_with_retry("${ROARING_FORMATSPEC_BASE}/testdata/bitmapwithoutruns.bin"
                                 "${CUCO_ROARING_DATA_DIR}/bitmapwithoutruns.bin"
                                 "d719ae2e0150a362ef7cf51c361527585891f01460b1a92bcfb6a7257282a442")

rapids_cmake_download_with_retry("${ROARING_FORMATSPEC_BASE}/testdata/bitmapwithruns.bin"
                                 "${CUCO_ROARING_DATA_DIR}/bitmapwithruns.bin"
                                 "1f1909bfdd354fa2f0694fe88b8076833ca5383ad9fc3f68f2709c84a2ab70e3")

rapids_cmake_download_with_retry("${ROARING_FORMATSPEC_BASE}/testdata64/portable_bitmap64.bin"
                                 "${CUCO_ROARING_DATA_DIR}/portable_bitmap64.bin"
                                 "b5a553a759167f5f9ccb3fa21552d943b4c73235635b753376f4faf62067d178")

message(STATUS "Roaring Bitmap test data downloaded to: ${CUCO_ROARING_DATA_DIR}")

# Define macro only when data is available
add_compile_definitions(CUCO_ROARING_DATA_DIR="${CUCO_ROARING_DATA_DIR}")