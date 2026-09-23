#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

# Preload WarpCore's dependencies so its nested CPM calls cannot resolve mutable branches.
CPMAddPackage(
  NAME helpers
  GIT_REPOSITORY https://gitlab.rlp.net/pararch/hpc_helpers.git
  GIT_TAG 232b928cd69bc29ad3d9c65d661bd1073cf4b779
  OPTIONS
    "HELPERS_BUILD_EXAMPLES OFF"
    "HELPERS_BUILD_TESTS OFF")

CPMAddPackage(
  NAME kiss_rng
  GIT_REPOSITORY https://github.com/sleeepyjack/kiss_rng.git
  GIT_TAG 9c6859ec6ffaa2a853636dbda4fc891e40737cd2
  OPTIONS
    "KISS_RNG_BUILD_EXAMPLES OFF")

CPMAddPackage(
  NAME warpcore
  GITHUB_REPOSITORY sleeepyjack/warpcore
  GIT_TAG 1a2fe03d438e8dfa1dbd8d9da73f183bd2d72051
  OPTIONS
    "WARPCORE_BUILD_TESTS OFF"
    "WARPCORE_BUILD_BENCHMARKS OFF"
    "WARPCORE_BUILD_EXAMPLES OFF")

ConfigureBench(
  WARPCORE_BLOOM_FILTER_BENCH
  warpcore_add_bench.cu
  warpcore_contains_bench.cu)
target_link_libraries(WARPCORE_BLOOM_FILTER_BENCH PRIVATE warpcore)
