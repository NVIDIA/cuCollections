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

#include <cuco/detail/trie/bit_vector/bit_vector_ref.cuh>
#include <cuco/storage.cuh>
#include <cuco/utility/allocator.hpp>

#include <thrust/device_vector.h>

namespace cuco {
namespace experimental {

/**
 * @brief Struct to store ranks of bits at 256-bit intervals
 */
struct rank {
  uint32_t abs_hi_;                    ///< Upper 32 bits of base
  uint8_t abs_lo_;                     ///< Lower 8 bits of base
  cuda::std::array<uint8_t, 3> rels_;  ///< Offsets for 64-bit sub-intervals

  /**
   * @brief Gets base rank of current 256-bit interval
   *
   * @return The base rank
   */
  __host__ __device__ uint64_t constexpr abs() const noexcept
  {
    return (static_cast<uint64_t>(abs_hi_) << 8) | abs_lo_;
  }

  /**
   * @brief Sets base rank of current 256-bit interval
   *
   * @param abs Base rank
   */
  __host__ __device__ void set_abs(uint64_t abs) noexcept
  {
    abs_hi_ = static_cast<uint32_t>(abs >> 8);
    abs_lo_ = static_cast<uint8_t>(abs);
  }
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
 * @tparam Allocator Type of allocator used for device storage
 */
template <class Allocator = cuco::cuda_allocator<std::byte>>
class bit_vector {
 public:
  /**
   * @brief Constructs an empty bitvector
   *
   * @param allocator Allocator for internal storage
   */
  bit_vector(Allocator const& allocator = Allocator{});
  ~bit_vector();

  /**
   * @brief adds a new bit at the end
   *
   * Grows internal storage if needed
   *
   * @param bit Boolean value of new bit to be added
   */
  void append(bool bit) noexcept;

  using size_type = std::size_t;  ///< size type to specify bit index
  /**
   * @brief Modifies a single bit
   *
   * @param index position of bit to be modified
   * @param bit new value of bit
   */
  void set(size_type index, bool bit) noexcept;

  /**
   * @brief Sets last bit to specified value
   *
   * @param bit new value of last bit
   */
  void set_last(bool bit) noexcept;

  /**
   * @brief Builds indexes for rank and select
   *
   * Also creates device-side snapshot
   */
  void build() noexcept;

  /**
   * @brief Bulk get operation
   *
   * @tparam KeyIt Device-accessible iterator to keys
   * @tparam OutputIt Device-accessible iterator to outputs
   *
   * @param keys_begin Begin iterator to keys list whose values are queried
   * @param keys_end End iterator to keys list
   * @param outputs_begin Begin iterator to outputs of get operation
   * @param stream Stream to execute get kernel
   */
  template <typename KeyIt, typename OutputIt>
  void get(KeyIt keys_begin,
           KeyIt keys_end,
           OutputIt outputs_begin,
           cuda_stream_ref stream = {}) const noexcept;

  /**
   * @brief Bulk set operation
   *
   * @tparam KeyIt Device-accessible iterator to keys
   * @tparam ValueIt Device-accessible iterator to values
   *
   * @param keys_begin Begin iterator to keys that need to modified
   * @param keys_end End iterator to keys
   * @param vals_begin Begin iterator to new bit values
   * @param stream Stream to execute set kernel
   */
  template <typename KeyIt, typename ValueIt>
  void set(KeyIt keys_begin,
           KeyIt keys_end,
           ValueIt vals_begin,
           cuda_stream_ref stream = {}) const noexcept;

  using allocator_type = Allocator;  ///< Allocator type
  using slot_type      = uint64_t;   ///< Slot type

  using words_storage_type =
    aow_storage<slot_type, 1, extent<size_type>, allocator_type>;  ///< storage type for words
  using ranks_storage_type =
    aow_storage<rank, 1, extent<size_type>, allocator_type>;  ///< storage type for ranks
  using selects_storage_type =
    aow_storage<size_type, 1, extent<size_type>, allocator_type>;  ///< storage type for selects

  /**
   *@brief Struct to hold all storage refs needed by bitvector_ref
   */
  struct device_storage_ref {
    using size_type       = size_type;     ///< Size type
    using slot_type       = slot_type;     ///< Slot type
    using bit_vector_type = bit_vector<>;  ///< bit_vector_ref needs this to access words_per_block

    typename words_storage_type::ref_type words_ref_;  ///< Words ref

    typename ranks_storage_type::ref_type ranks_ref_;      ///< Ranks refs
    typename selects_storage_type::ref_type selects_ref_;  ///< Selects refs

    typename ranks_storage_type::ref_type ranks0_ref_;      ///< Ranks refs for 0 bits
    typename selects_storage_type::ref_type selects0_ref_;  ///< Selects refs 0 bits
  };

  template <typename... Operators>
  using ref_type =
    bit_vector_ref<device_storage_ref, Operators...>;  ///< Non-owning container ref type

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
  size_type constexpr size() const noexcept { return n_bits_; }

  static constexpr size_type words_per_block = 4;  ///< Tradeoff between space efficiency and perf.

 private:
  size_type n_bits_;  ///< Number of bits bit_vector currently holds

  // These could be public if needed by other classes. Private for now
  static constexpr size_type bits_per_word  = sizeof(slot_type) * 8;            ///< Bits in a word
  static constexpr size_type bits_per_block = words_per_block * bits_per_word;  ///< Trivial

  // Host-side structures
  std::vector<slot_type> words_;  ///< Words vector that represents all bits

  // Device-side structures
  thrust::device_vector<slot_type> d_words_;   ///< Device words vector
  thrust::device_vector<rank> ranks_;          ///< Rank values for every 256-th bit (4-th word)
  thrust::device_vector<rank> ranks0_;         ///< Same as ranks_ but for `0` bits
  thrust::device_vector<size_type> selects_;   ///< Block indices of (0, 256, 512...)th `1` bit
  thrust::device_vector<size_type> selects0_;  ///< Same as selects_, but for `0` bits

  allocator_type allocator_;  ///< Allocator used to (de)allocate temporary storage
  std::unique_ptr<words_storage_type> aow_words_;  ///< Array of window storage structure
  std::unique_ptr<ranks_storage_type> aow_ranks_;
  std::unique_ptr<ranks_storage_type> aow_ranks0_;
  std::unique_ptr<selects_storage_type> aow_selects_;
  std::unique_ptr<selects_storage_type> aow_selects0_;

  /**
   * @brief Populates rank and select indexes on device
   *
   * @param ranks Output array of ranks
   * @param selects Output array of selects
   * @param flip_bits If true, negate bits to construct indexes for `0` bits
   */
  void build_ranks_and_selects(thrust::device_vector<rank>& ranks,
                               thrust::device_vector<size_type>& selects,
                               bool flip_bits);

  /**
   * @brief Creates a new window structure on device and initializes it from a device array
   *
   * @tparam T Type of device array elements
   * @tparam storage_type Storage type
   *
   * @param aow pointer to destination (device window structure)
   * @param device_array device array whose contents are used to intialize aow
   */
  template <class T, class storage_type>
  void copy_device_array_to_aow(std::unique_ptr<storage_type>* aow,
                                thrust::device_vector<T>& device_array) noexcept;

  /**
   * @brief Helper function to calculate grid size for simple kernels
   *
   * @param num_elements Elements being processed by kernel
   *
   * @return grid size
   */
  size_type constexpr default_grid_size(size_type num_elements) const noexcept
  {
    return (num_elements - 1) / (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE) + 1;
  }
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/trie/bit_vector/bit_vector.inl>
