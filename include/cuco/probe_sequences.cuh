/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuco/detail/probe_sequence_impl.cuh>

namespace cuco {

template <uint32_t CGSize>
class probe_sequence_base {
 public:
  /**
   * @brief Returns the size of the CUDA cooperative thread group.
   */
  static constexpr std::size_t cg_size = CGSize;

  /**
   * @brief Returns the number of elements loaded with each vector load.
   */
  static constexpr uint32_t vector_width() noexcept { return 2u; }
};

template <uint32_t CGSize, typename Hash>
class linear_probing : public probe_sequence_base<CGSize> {
 public:
  using probe_sequence_base<CGSize>::cg_size;
  using probe_sequence_base<CGSize>::vector_width;

  template <typename Key, typename Value, cuda::thread_scope Scope>
  using impl = detail::linear_probing_impl<Key, Value, Scope, vector_width(), CGSize, Hash>;
};

template <uint32_t CGSize, typename Hash1, typename Hash2>
class double_hashing : public probe_sequence_base<CGSize> {
 public:
  using probe_sequence_base<CGSize>::cg_size;
  using probe_sequence_base<CGSize>::vector_width;

  template <typename Key, typename Value, cuda::thread_scope Scope>
  using impl = detail::double_hashing_impl<Key, Value, Scope, vector_width(), CGSize, Hash1, Hash2>;
};

}  // namespace cuco
