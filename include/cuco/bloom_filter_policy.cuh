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

#include <cuco/detail/bloom_filter/bloom_filter_policy.cuh>
#include <cuco/hash_functions.cuh>

#include <cstdint>

namespace cuco {

/**
 * @brief Sectorized Bloom filter policy with multiplicative-hashing fingerprint generation.
 *
 * Implements the Sectorized Bloom Filter (SBF) variant from "Optimizing Bloom Filters for Modern
 * GPU Architectures" (arXiv:2512.15595).
 *
 * Requires a 64-bit hash function: the result is split into upper 32 bits (block selection via
 * multiply-shift) and lower 32 bits (pattern generation).
 *
 * @tparam Key Key type to hash.
 * @tparam Hash 64-bit hash functor type. Defaults to `cuco::xxhash_64<Key>`.
 * @tparam Word Underlying word type of a filter block. Defaults to `std::uint32_t`.
 * @tparam WordsPerBlock Words per filter block. Defaults to the number of `Word`s that fit in one
 * 32-byte sector.
 * @tparam PatternBits Fingerprint bits per key (paper's k). Defaults to `WordsPerBlock`.
 * @tparam AddHorizontalLayout CG size for add (paper's Theta). Defaults to `WordsPerBlock` for
 * fully horizontal add.
 * @tparam AddVerticalLayout Words per thread per add step (paper's Phi). Defaults to `1` for fully
 * horizontal add.
 * @tparam ContainsHorizontalLayout CG size for contains. Defaults to `1` for fully vertical
 * contains.
 * @tparam ContainsVerticalLayout Words per thread per contains step. Defaults to `WordsPerBlock`
 * for fully vertical contains.
 * @tparam ConditionalAdd When `true`, `add` reads each word before the atomic OR and skips the
 * write when the required bits are already set. Trades a read for fewer atomic writes; beneficial
 * when the filter is highly contended (e.g. close to full) or the input has many duplicate keys.
 * @tparam EarlyExitContains When `true`, `contains` short-circuits a thread's evaluation on the
 * first missing fingerprint slice. Beneficial when queried keys have a low match rate and filter
 * contention is low.
 * @tparam PersistingL2Access When `true`, annotates global-memory filter accesses with an L2
 * persisting access policy. Reserve/reset persisting L2
 * separately (e.g. `cudaDeviceSetLimit`). Enable only when the working set fits the reserved
 * region, otherwise persisting lines can thrash the cache and slow other work.
 */
template <class Key,
          class Hash                             = cuco::xxhash_64<Key>,
          class Word                             = std::uint32_t,
          std::uint32_t WordsPerBlock            = 32 / sizeof(Word),
          std::uint32_t PatternBits              = WordsPerBlock,
          std::uint32_t AddHorizontalLayout      = WordsPerBlock,
          std::uint32_t AddVerticalLayout        = 1,
          std::uint32_t ContainsHorizontalLayout = 1,
          std::uint32_t ContainsVerticalLayout   = WordsPerBlock,
          bool ConditionalAdd                    = false,
          bool EarlyExitContains                 = false,
          bool PersistingL2Access                = false>
using bloom_filter_policy = detail::bloom_filter_policy<Hash,
                                                        Word,
                                                        WordsPerBlock,
                                                        PatternBits,
                                                        AddHorizontalLayout,
                                                        AddVerticalLayout,
                                                        ContainsHorizontalLayout,
                                                        ContainsVerticalLayout,
                                                        ConditionalAdd,
                                                        EarlyExitContains,
                                                        PersistingL2Access>;

}  // namespace cuco
