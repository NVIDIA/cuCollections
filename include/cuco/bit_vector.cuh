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

/**
 * @brief Struct to store ranks of bits at 256-bit intervals
 */
struct rank {
  uint32_t abs_hi_;  ///< Upper 32 bits of base
  uint8_t abs_lo_;   ///< Lower 8 bits of base
  uint8_t rels_[3];  ///< Four offsets for 64-bit sub-intervals

  /**
   * @brief Gets base rank of current 256-bit interval
   *
   * @return The base rank
   */
  __host__ __device__ uint64_t abs() const
  {
    return (static_cast<uint64_t>(abs_hi_) << 8) | abs_lo_;
  }

  /**
   * @brief Sets base rank of current 256-bit interval
   *
   * @param abs Base rank
   */
  void set_abs(uint64_t abs)
  {
    abs_hi_ = static_cast<uint32_t>(abs >> 8);
    abs_lo_ = static_cast<uint8_t>(abs);
  }
};
/**
 * @brief Union of 64-bit word with rank
 *
 * Need this so that all aow_storage structures in bitvector have 64-bit element type
 */
union rank_union {
  uint64_t word_;  ///< word view
  rank rank_;      ///< rank view
};

/**
 * @brief Bitvector class with rank and select index structures
 *
 * In addition to standard bitvector get/set operations, this class provides
 * rank and select operation API. It maintains index structures to make both these
 * new operations close to constant time.
 * Bitvector construction happens on host, after which the structures are moved to device.
 * All subsequent read-only operations access device structures only.
 *
 * @tparam Key Type of the index that specifies which bit to access/modify
 * @tparam Extent Data structure size type
 * @tparam Scope The scope in which operations will be performed by individual threads.
 * @tparam Allocator Type of allocator used for device storage
 * @tparam Storage Slot window storage type
 */

template <class Key                = uint64_t,
          class Extent             = cuco::experimental::extent<std::size_t>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Allocator          = cuco::cuda_allocator<std::byte>,
          class Storage            = cuco::experimental::aow_storage<1>>
class bit_vector {
 public:
  bit_vector();
  ~bit_vector();

  /**
   * @brief adds a new bit at the end
   *
   * @param bit Boolean value of new bit to be added
   */
  void add(bool bit);

  /**
   * @brief Builds indexes for rank and select
   *
   * Also creates device-side snapshot
   */
  void build();

  /**
   * @brief Modifies a single bit
   *
   * @param key position of bit to be modified
   * @param bit new value of bit
   */
  void set(Key key, bool bit);

  /**
   * @brief Sets last bit to specified value
   *
   * @param bit new value of last bit
   */
  void set_last(bool bit);

  static constexpr auto cg_size      = 1;      ///< CG size used to for probing
  static constexpr auto window_size  = 1;      ///< Window size used to for probing
  static constexpr auto thread_scope = Scope;  ///< CUDA thread scope

  using extent_type =
    decltype(make_valid_extent<cg_size, window_size>(std::declval<Extent>()));  ///< Extent type
  using allocator_type = Allocator;                                             ///< Allocator type
  using storage_type =
    detail::storage<Storage, Key, extent_type, allocator_type>;  ///< Storage type

  using storage_ref_type = typename storage_type::ref_type;  ///< Non-owning window storage ref type
  template <typename... Operators>
  using ref_type =
    cuco::experimental::bit_vector_ref<storage_ref_type,
                                       Operators...>;  ///< Non-owning container ref type

  /**
   * @brief Get device ref with operators.
   *
   * @tparam Operators Set of `cuco::op` to be provided by the ref
   *
   * @param ops List of operators, e.g., `cuco::bv_read`
   *
   * @return Device ref of the current `bit_vector` object
   */
  template <typename... Operators>
  [[nodiscard]] auto ref(Operators... ops) const noexcept;

  /**
   * @brief Get the number of bits bit_vector holds
   *
   * @return Number of bits bit_vector holds
   */
  size_t size() const { return n_bits_; }

 private:
  uint64_t n_bits_;  ///< Number of bits added to bit_vector

  // Host-side structures
  std::vector<uint64_t> words_;     ///< Words vector that represents all bits
  std::vector<rank> ranks_;         ///< Holds the rank values for every 256-th bit (4-th word)
  std::vector<rank> ranks0_;        ///< Same as ranks_ but for `0` bits
  std::vector<uint64_t> selects_;   ///< Holds pointers to (0, 256, 512...)th `1` bit in ranks_
  std::vector<uint64_t> selects0_;  ///< Same as selects_, but for `0` bits

  // Device-side structures
  allocator_type allocator_;  ///< Allocator used to (de)allocate temporary storage
  storage_type *aow_words_, *aow_ranks_, *aow_selects_, *aow_ranks0_, *aow_selects0_;

  /**
   * @brief Creates a new window structure on device and intitializes it with contents of host array
   *
   * @tparam T Type of host array elements
   *
   * @param aow pointer to destination (device window structure)
   * @param host_array host array whose contents are used to intialize aow
   */
  template <class T>
  void copy_host_array_to_aow(storage_type** aow, std::vector<T>& host_array);

  /**
   * @brief Constructs device-side structures and clears host-side structures
   *
   * Effectively takes a snapshot of the bitvector and creates a device-side copy
   */
  void move_to_device();
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/bit_vector/bit_vector.inl>
#include <cuco/detail/bit_vector/bit_vector_ref.inl>
