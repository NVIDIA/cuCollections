/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(CUCO_DEBUG)

#include <cstdio>

#if defined(NDEBUG)
#define CUCO_DETAIL_RESTORE_NDEBUG
#undef NDEBUG
#endif

#include <assert.h>

namespace cuco::detail {

[[noreturn]] __device__ inline void debug_assert_fail(char const* message,
                                                      char const* file,
                                                      int line) noexcept
{
  printf("cuco assertion failed: %s (%s:%d)\n", message, file, line);
  assert(false);
}

}  // namespace cuco::detail

#if defined(CUCO_DETAIL_RESTORE_NDEBUG)
#define NDEBUG
#include <assert.h>
#undef CUCO_DETAIL_RESTORE_NDEBUG
#endif

#define CUCO_DEBUG_ASSERT(condition, message) \
  ((condition) ? static_cast<void>(0)         \
               : ::cuco::detail::debug_assert_fail((message), __FILE__, __LINE__))

#else

#define CUCO_DEBUG_ASSERT(condition, message) static_cast<void>(0)

#endif
