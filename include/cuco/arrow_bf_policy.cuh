/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuco/detail/bloom_filter/arrow_bf_policy_impl.cuh>

#include <cstdint>

namespace cuco {

/**
 * @brief A policy that defines how a Blocked Bloom Filter generates and stores a key's fingerprint.
 *
 * @note `Word` type must be an atomically updatable integral type. `WordsPerBlock` must
 * be a power-of-two.
 *
 * @tparam Hash Hash function used to generate a key's fingerprint
 * @tparam Word Underlying word/segment type of a filter block
 * @tparam WordsPerBlock Number of words/segments in each block
 */
template <class T>
class arrow_bf_policy {
  using impl_type = cuco::detail::arrow_bf_policy_impl<T>;

 public:
  using hasher             = typename impl_type::hasher;              ///< Type of the hash function
  using hash_argument_type = typename impl_type::hash_argument_type;  ///< Hash function input type
  using hash_result_type   = typename impl_type::hash_result_type;    ///< hash function output type
  using word_type =
    typename impl_type::word_type;  ///< Underlying word/segment type of a filter block

  static constexpr std::uint32_t words_per_block =
    impl_type::words_per_block;  ///< Number of words/segments in each filter block

  static constexpr std::uint32_t bits_set_per_block =
    impl_type::bits_set_per_block;  ///< Number of words/segments in each filter block

 public:
  /**
   * @brief Constructs the `arrow_bf_policy` object.
   *
   * @MH: Fix this doc.
   *
   * @throws Compile-time error if the specified number of words in a filter block is not a
   * power-of-two or is larger than 32. If called from host: throws exception; If called from
   * device: Traps the kernel.
   *
   * @throws If the `hash_result_type` is too narrow to generate the requested number of
   * `pattern_bits`. If called from host: throws exception; If called from device: Traps the kernel.
   *
   * @throws If `pattern_bits` is smaller than the number of words in a filter block or larger than
   * the total number of bits in a filter block. If called from host: throws exception; If called
   * from device: Traps the kernel.
   *
   * @param num_blocks Number of bloom filter blocks
   * @param hash Hash function used to generate a key's fingerprint
   */
  __host__ __device__ constexpr arrow_bf_policy(std::uint32_t num_blocks, hasher hash = {});

  /**
   * @brief Generates the hash value for a given key.
   *
   * @note This function is meant as a customization point and is only used in the internals of the
   * `bloom_filter(_ref)` implementation.
   *
   * @param key The key to hash
   *
   * @return The hash value of the key
   */
  __device__ constexpr hash_result_type hash(hash_argument_type const& key) const;

  /**
   * @brief Determines the filter block a key is added into.
   *
   * @note This function is meant as a customization point and is only used in the internals of the
   * `bloom_filter(_ref)` implementation.
   *
   * @tparam Extent Size type that is used to determine the number of blocks in the filter
   *
   * @param hash Hash value of the key
   * @param num_blocks Number of block in the filter
   *
   * @return The block index for the given key's hash value
   */
  template <class Extent>
  __device__ constexpr auto block_index(hash_result_type hash, Extent num_blocks) const;

  /**
   * @brief Determines the fingerprint pattern for a word/segment within the filter block for a
   * given key's hash value.
   *
   * @note This function is meant as a customization point and is only used in the internals of the
   * `bloom_filter(_ref)` implementation.
   *
   * @param hash Hash value of the key
   * @param word_index Target word/segment within the filter block
   *
   * @return The bit pattern for the word/segment in the filter block
   */
  __device__ constexpr word_type word_pattern(hash_result_type hash,
                                              std::uint32_t word_index) const;

 private:
  impl_type impl_;  ///< Policy implementation
};

}  // namespace cuco

#include <cuco/detail/bloom_filter/arrow_bf_policy.inl>