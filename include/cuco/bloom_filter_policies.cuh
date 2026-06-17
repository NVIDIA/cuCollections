/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include <cuco/detail/bloom_filter/parametric_filter_policy.cuh>
#include <cuco/hash_functions.cuh>

#include <cstdint>

namespace cuco {

/**
 * @brief Sectorized Bloom filter policy with multiplicative-hashing fingerprint generation.
 *
 * Implements the Sectorized Bloom Filter (SBF) variant from "Optimizing Bloom Filters for Modern
 * GPU Architectures" (arXiv:2512.15595). Distributes `PatternBits` set bits across `WordsPerBlock`
 * words via compile-time salt-based multiplicative hashing.
 *
 * Requires a 64-bit hash function: the result is split into upper 32 bits (block selection via
 * multiply-shift) and lower 32 bits (pattern generation). This is a permanent design requirement.
 *
 * @note Constructor signature: `parametric_filter_policy(Hash hash = {})`.
 *
 * @tparam Hash 64-bit hash functor.
 * @tparam Word Underlying word type of a filter block.
 * @tparam WordsPerBlock Words per filter block.
 * @tparam PatternBits Fingerprint bits per key (paper's k).
 * @tparam AddHorizontalLayout CG size for add (paper's Theta).
 * @tparam AddVerticalLayout Words per thread per add step (paper's Phi).
 * @tparam ContainsHorizontalLayout CG size for contains.
 * @tparam ContainsVerticalLayout Words per thread per contains step.
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

/**
 * @brief Default Bloom filter policy used by `cuco::bloom_filter` when no policy is specified.
 *
 * Alias for a `parametric_filter_policy` instantiation with paper-recommended layouts on a 256-bit
 * block: 8 x `uint32_t` words, 8 fingerprint bits per key, fully horizontal add (Theta=8) and fully
 * vertical contains (Phi=8).
 *
 * @tparam Key The key type to generate a fingerprint for.
 * @tparam XXHash64 64-bit XXHash functor template. Defaults to `cuco::xxhash_64`.
 */
template <class Key, template <typename> class XXHash64 = cuco::xxhash_64>
using default_filter_policy =
  parametric_filter_policy<XXHash64<Key>, std::uint32_t, 8, 8, 8, 1, 1, 8>;

}  // namespace cuco
