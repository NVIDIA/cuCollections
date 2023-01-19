/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuco/detail/storage.cuh>

namespace cuco {
namespace experimental {
/**
 * @brief Public array of windows storage class.
 *
 * @tparam WindowSize Number of elements per window storage
 */
template <int32_t WindowSize>
class aow_storage {
 public:
  /// Number of elements per window storage
  static constexpr int32_t window_size = WindowSize;

  /// Type of implementation details
  template <class T, class Extent, class Allocator>
  using impl = detail::aow_storage<window_size, T, Extent, Allocator>;
};

}  // namespace experimental
}  // namespace cuco
