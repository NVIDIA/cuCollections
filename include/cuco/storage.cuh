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

#include <cuco/detail/storage/storage.cuh>

namespace cuco {
namespace experimental {

/**
 * @brief Window data structure type
 *
 * @tparam Key The type of the window elements
 * @tparam WindowSize Number of elements per window
 */
template <typename T, int32_t WindowSize>
struct window : public cuda::std::array<T, WindowSize> {
 public:
  static int32_t constexpr window_size = WindowSize;  ///< Number of elements per window
};

/**

 * @brief Public Array of slot Windows storage class.
 *
 * The window size defines the workload granularity for each CUDA thread, i.e., how many slots a
 * thread would concurrently operate on when performing modify or lookup operations. cuCollections
 * uses the AoW storage to supersede the raw flat slot storage due to its superior granularity
 * control: When window size equals one, AoW performs the same as the flat storage. If the
 * underlying operation is more memory bandwidth bound, e.g., high occupancy multimap operations, a
 * larger window size can reduce the length of probing sequences thus improve runtime performance.
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
