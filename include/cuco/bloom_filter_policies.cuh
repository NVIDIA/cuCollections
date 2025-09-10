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

#include <cuco/detail/bloom_filter/arrow_filter_policy.cuh>
#include <cuco/detail/bloom_filter/default_filter_policy.cuh>
#include <cuco/detail/bloom_filter/parametric_filter_policy.cuh>
#include <cuco/hash_functions.cuh>

#include <cstdint>

namespace cuco {

/**
 * @brief A policy that defines how Arrow Block-Split Bloom Filter generates and stores a key's
 * fingerprint.
 *
 * @note Constructor signature: `arrow_filter_policy(hasher hash = {})`
 *
 * @tparam Key The type of the values to generate a fingerprint for.
 * @tparam XXHash64 Custom (64 bit) XXHash hasher to generate a key's fingerprint.
 * By default, cuco::xxhash_64 hasher will be used.
 *
 */
template <class Key, template <typename> class XXHash64 = cuco::xxhash_64>
using arrow_filter_policy = detail::arrow_filter_policy<Key, XXHash64>;

/**
 * @brief The default policy that defines how a Blocked Bloom Filter generates and stores a key's
 * fingerprint.
 *
 * @note `Word` type must be an atomically updatable integral type. `WordsPerBlock` must
 * be a power-of-two.
 *
 * @note Constructor signature: `default_filter_policy(uint32_t pattern_bits = words_per_block, Hash
 * hash = {})`
 *
 * @tparam Hash Hash function used to generate a key's fingerprint
 * @tparam Word Underlying word/segment type of a filter block
 * @tparam WordsPerBlock Number of words/segments in each block
 */
template <class Hash, class Word, std::uint32_t WordsPerBlock>
using default_filter_policy = detail::default_filter_policy<Hash, Word, WordsPerBlock>;

namespace experimental {

/**
 * @brief A parametric Bloom filter policy that specifies the vectorization layout and the number of
 * bits in the key's fingerprint.
 *
 * @note `Word` type must be an atomically updatable integral type. `WordsPerBlock` must
 * be a power-of-two.
 *
 * @note Constructor signature: `parametric_filter_policy(Hash hash = {})`
 *
 * @tparam Hash Hash function used to generate a key's fingerprint
 * @tparam Word Underlying word/segment type of a filter block
 * @tparam WordsPerBlock Number of words/segments in each block
 * @tparam PatternBits Number of bits in the key's fingerprint
 * @tparam AddHorizontalLayout Horizontal vectorization layout for add operation
 * @tparam AddVerticalLayout Vertical vectorization layout for add operation
 * @tparam ContainsHorizontalLayout Horizontal vectorization layout for contains operation
 * @tparam ContainsVerticalLayout Vertical vectorization layout for contains operation
 */
template <class Hash,
          class Word,
          std::uint32_t WordsPerBlock,
          std::uint32_t PatternBits,
          std::uint32_t AddHorizontalLayout,
          std::uint32_t AddVerticalLayout,
          std::uint32_t ContainsHorizontalLayout,
          std::uint32_t ContainsVerticalLayout>
using parametric_filter_policy = detail::parametric_filter_policy<Hash,
                                                                  Word,
                                                                  WordsPerBlock,
                                                                  PatternBits,
                                                                  AddHorizontalLayout,
                                                                  AddVerticalLayout,
                                                                  ContainsHorizontalLayout,
                                                                  ContainsVerticalLayout>;
}  // namespace experimental

}  // namespace cuco