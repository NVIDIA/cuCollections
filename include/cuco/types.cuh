/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/utility/strong_type.cuh>

/**
 * @brief Defines various strong type wrappers used across this library.
 *
 * @note Each strong type inherits from `cuco::detail::strong_type<T>`. `CUCO_DEFINE_STRONG_TYPE`
 * and `CUCO_DEFINE_TEMPLATE_STRONG_TYPE` are convenience macros used to define a named type in a
 * single line, e.g., `CUCO_DEFINE_STRONG_TYPE(foo, double)` defines `struct foo : public
 * cuco::detail::strong_type<double> {...};`, where `cuco::foo{42.0}` is implicitly convertible to
 * `double{42.0}`.
 */

namespace cuco {
/**
 * @brief A strong type wrapper `cuco::empty_key<Key>` used to denote the empty key sentinel.
 */
CUCO_DEFINE_TEMPLATE_STRONG_TYPE(empty_key);

/**
 * @brief A strong type wrapper `cuco::empty_value<T>` used to denote the empty value sentinel.
 */
CUCO_DEFINE_TEMPLATE_STRONG_TYPE(empty_value);

/**
 * @brief A strong type wrapper `cuco::erased_key<Key>` used to denote the erased key sentinel.
 */
CUCO_DEFINE_TEMPLATE_STRONG_TYPE(erased_key);

}  // namespace cuco
