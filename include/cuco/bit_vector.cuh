/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuco/bit_vector_ref.cuh>
#include <cuco/extent.cuh>
#include <cuco/storage.cuh>
#include <cuco/utility/allocator.hpp>

#include <thrust/device_vector.h>

#include <cuda/atomic>

namespace cuco {
namespace experimental {

struct rank {
  // Basically a uint64_t split into 1 uin32_t and 2 uint8_t
  uint32_t abs_hi_;
  uint8_t abs_lo_;
  uint8_t rels_[3];

  __host__ __device__ uint64_t abs() const { return ((uint64_t)abs_hi_ << 8) | abs_lo_; }
  void set_abs(uint64_t abs)
  {
    abs_hi_ = (uint32_t)(abs >> 8);
    abs_lo_ = (uint8_t)abs;
  }
};

// Need this union to use uint64_t for all aow_storage structures
union rank_union {
  uint64_t word_;
  rank rank_;
};

template <class Key                = uint64_t,
          class Extent             = cuco::experimental::extent<std::size_t>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Allocator          = cuco::cuda_allocator<std::byte>,
          class Storage            = cuco::experimental::aow_storage<1>>
class bit_vector {
 public:
  bit_vector();
  ~bit_vector();

  void add(bool bit);  // adds a new bit at the end
  void build();        // builds indexes for rank and select.

  void set(Key key, bool bit);
  void set_last(bool bit);

  static constexpr auto cg_size      = 1;
  static constexpr auto window_size  = 1;
  static constexpr auto thread_scope = Scope;

  using key_type       = Key;  ///< Key type
  using value_type     = Key;  ///< Key type
  using extent_type    = decltype(make_valid_extent<cg_size, window_size>(std::declval<Extent>()));
  using size_type      = typename extent_type::value_type;  ///< Size type
  using allocator_type = Allocator;                         ///< Allocator type
  using storage_type =
    detail::storage<Storage, value_type, extent_type, allocator_type>;  ///< Storage type

  using storage_ref_type = typename storage_type::ref_type;  ///< Non-owning window storage ref type
  template <typename... Operators>
  using ref_type =
    cuco::experimental::bit_vector_ref<storage_ref_type,
                                       Operators...>;  ///< Non-owning container ref type

  template <typename... Operators>
  [[nodiscard]] auto ref(Operators... ops) const noexcept;

  size_t size() const { return n_bits_; }

 private:
  uint64_t n_bits_;

  // Host structures
  std::vector<uint64_t> words_;
  std::vector<rank> ranks_, ranks0_;
  std::vector<uint64_t> selects_, selects0_;

  // Device structures
  allocator_type allocator_;  ///< Allocator used to (de)allocate temporary storage
  storage_type *aow_words_, *aow_ranks_, *aow_selects_, *aow_ranks0_, *aow_selects0_;

  template <class T>
  void copy_host_array_to_aow(storage_type** aow, std::vector<T>& host_array);

  void move_to_device();
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/bit_vector/bit_vector.inl>
#include <cuco/detail/bit_vector/bit_vector_ref.inl>
