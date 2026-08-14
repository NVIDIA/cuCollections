/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 * @tparam WordBytes Size in bytes of the underlying word type. Must be `4` or `8`. Defaults to `4`.
 * @tparam WordsPerBlock Words per filter block. Defaults to the number of words that fit in one
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
          std::uint32_t WordBytes                = 4,
          std::uint32_t WordsPerBlock            = 32 / WordBytes,
          std::uint32_t PatternBits              = WordsPerBlock,
          std::uint32_t AddHorizontalLayout      = WordsPerBlock,
          std::uint32_t AddVerticalLayout        = 1,
          std::uint32_t ContainsHorizontalLayout = 1,
          std::uint32_t ContainsVerticalLayout   = WordsPerBlock,
          bool ConditionalAdd                    = false,
          bool EarlyExitContains                 = false,
          bool PersistingL2Access                = false>
using bloom_filter_policy = detail::bloom_filter_policy<Hash,
                                                        WordBytes,
                                                        WordsPerBlock,
                                                        PatternBits,
                                                        AddHorizontalLayout,
                                                        AddVerticalLayout,
                                                        ContainsHorizontalLayout,
                                                        ContainsVerticalLayout,
                                                        ConditionalAdd,
                                                        EarlyExitContains,
                                                        PersistingL2Access>;

/**
 * @brief Deprecated compatibility alias for the old parametric Bloom filter policy API.
 *
 * The supplied `Word` type is used only to select the internal word width through `sizeof(Word)`.
 *
 * @note This alias should not be used in new code and may be removed on short notice. Use
 * `cuco::bloom_filter_policy` directly instead.
 *
 * @tparam Hash 64-bit hash functor type.
 * @tparam Word Type whose size selects the underlying word width.
 * @tparam WordsPerBlock Words per filter block.
 * @tparam PatternBits Fingerprint bits per key (paper's k).
 * @tparam AddHorizontalLayout CG size for add (paper's Theta).
 * @tparam AddVerticalLayout Words per thread per add step (paper's Phi).
 * @tparam ContainsHorizontalLayout CG size for contains.
 * @tparam ContainsVerticalLayout Words per thread per contains step.
 * @tparam ConditionalAdd Whether to skip redundant atomic writes.
 * @tparam EarlyExitContains Whether to short-circuit contains on the first missing slice.
 * @tparam PersistingL2Access Whether to annotate global-memory accesses as persisting.
 */
template <class Hash,
          class Word,
          std::uint32_t WordsPerBlock,
          std::uint32_t PatternBits,
          std::uint32_t AddHorizontalLayout,
          std::uint32_t AddVerticalLayout,
          std::uint32_t ContainsHorizontalLayout,
          std::uint32_t ContainsVerticalLayout,
          bool ConditionalAdd,
          bool EarlyExitContains,
          bool PersistingL2Access = false>
using parametric_filter_policy =
  detail::bloom_filter_policy<Hash,
                              static_cast<std::uint32_t>(sizeof(Word)),
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
