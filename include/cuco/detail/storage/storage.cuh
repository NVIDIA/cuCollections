/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuco/aow_storage.cuh>

namespace cuco {
namespace experimental {
namespace detail {
/**
 * @brief Intermediate class internally used by data structures
 *
 * @tparam StorageImpl Storage implementation class
 * @tparam T Storage element type
 * @tparam Extent Type of extent denoting number of windows
 * @tparam Allocator Type of allocator used for device storage
 */
template <class StorageImpl, class T, class Extent, class Allocator>
class storage : StorageImpl::template impl<T, Extent, Allocator> {
 public:
  /// Storage implementation type
  using impl_type      = typename StorageImpl::template impl<T, Extent, Allocator>;
  using ref_type       = typename impl_type::ref_type;        ///< Storage ref type
  using value_type     = typename impl_type::value_type;      ///< Storage value type
  using allocator_type = typename impl_type::allocator_type;  ///< Storage value type

  /// Number of elements per window
  static constexpr int window_size = impl_type::window_size;

  using impl_type::allocator;
  using impl_type::capacity;
  using impl_type::data;
  using impl_type::initialize;
  using impl_type::num_windows;
  using impl_type::ref;

  /**
   * @brief Constructs storage.
   *
   * @param size Number of slots to (de)allocate
   * @param allocator Allocator used for (de)allocating device storage
   */
  explicit constexpr storage(Extent size, Allocator const& allocator) : impl_type{size, allocator}
  {
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
