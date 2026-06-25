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

#include <cuco/detail/__config>
#include <cuco/detail/bloom_filter/kernels.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/utility/cuda.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/detail/utility/math.cuh>
#include <cuco/detail/utils.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuco/utility/traits.hpp>

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>  // TODO #include <cuda/std/algorithm> once available
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/stream_ref>
#include <cuda/utility>

#include <cooperative_groups.h>

#include <cstdint>

namespace cuco::detail {

template <class Key, class Extent, cuda::thread_scope Scope, class Policy>
class bloom_filter_impl {
 public:
  using key_type    = Key;
  using extent_type = Extent;
  using size_type   = typename extent_type::value_type;
  using policy_type = Policy;
  using word_type   = typename policy_type::word_type;
  static_assert(sizeof(word_type) == 4 || sizeof(word_type) == 8,
                "word_type must be 4 or 8 bytes wide for atomicOr");
  // atomicOr overloads resolve on canonical 32- and 64-bit unsigned integer types.
  // Normalize by size so any policy-provided word_type (uint32_t, uint64_t, unsigned long, ...)
  // resolves to a matching overload via the reinterpret_cast in atomic_or().
  using atomic_word_type =
    cuda::std::conditional_t<sizeof(word_type) == 8, unsigned long long, unsigned int>;

  static constexpr auto thread_scope    = Scope;
  static constexpr auto words_per_block = policy_type::words_per_block;

  static constexpr auto add_vertical_layout        = policy_type::add_vertical_layout;
  static constexpr auto add_horizontal_layout      = policy_type::add_horizontal_layout;
  static constexpr auto contains_vertical_layout   = policy_type::contains_vertical_layout;
  static constexpr auto contains_horizontal_layout = policy_type::contains_horizontal_layout;
  static constexpr bool conditional_add            = policy_type::conditional_add;
  static constexpr bool early_exit_contains        = policy_type::early_exit_contains;
  static constexpr auto add_loop_count =
    words_per_block / (add_vertical_layout * add_horizontal_layout);
  static constexpr auto contains_loop_count =
    words_per_block / (contains_vertical_layout * contains_horizontal_layout);

  static_assert(cuda::std::has_single_bit(words_per_block) and words_per_block <= 32,
                "Number of words per block must be a power-of-two and less than or equal to 32");
  static_assert(
    cuda::std::is_constructible_v<cuda::atomic_ref<word_type, Scope>, word_type&> &&
      cuda::std::is_invocable_r_v<word_type,
                                  decltype(&cuda::atomic_ref<word_type, Scope>::fetch_or),
                                  cuda::atomic_ref<word_type, Scope>*,
                                  word_type,
                                  cuda::std::memory_order>,
    "Invalid word type");

  __host__ __device__ static constexpr size_t alignment() noexcept
  {
    // Maximum alignment is 32 bytes which is equivalent to one sector
    return cuda::std::min(
      static_cast<size_t>(32),
      static_cast<size_t>(cuda::std::max(add_vertical_layout, contains_vertical_layout) *
                          sizeof(word_type)));
  }

  struct filter_block_type {
   private:
    alignas(alignment()) word_type data_[words_per_block];
  };

  __host__ __device__ explicit constexpr bloom_filter_impl(filter_block_type* filter,
                                                           Extent num_blocks,
                                                           cuda_thread_scope<Scope>,
                                                           Policy policy)
    : words_{reinterpret_cast<word_type*>(filter)}, num_blocks_{num_blocks}, policy_{policy}
  {
  }

  __host__ __device__ explicit constexpr bloom_filter_impl(word_type* filter,
                                                           Extent num_blocks,
                                                           cuda_thread_scope<Scope>,
                                                           Policy policy)
    : words_{filter}, num_blocks_{num_blocks}, policy_{policy}
  {
  }

  template <class CG>
  __device__ constexpr void clear(CG group)
  {
    for (int i = group.thread_rank(); i < static_cast<size_type>(num_blocks_) * words_per_block;
         i += group.size()) {
      words_[i] = 0;
    }
  }

  __host__ constexpr void clear(cuda::stream_ref stream)
  {
    this->clear_async(stream);
    stream.sync();
  }

  __host__ constexpr void clear_async(cuda::stream_ref stream)
  {
    CUCO_CUDA_TRY(cub::DeviceFor::ForEachN(
      words_,
      static_cast<size_type>(num_blocks_) * words_per_block,
      [] __device__(word_type & word) { word = 0; },
      stream.get()));
  }

  __host__ constexpr void merge(bloom_filter_impl<Key, Extent, Scope, Policy> const& other,
                                cuda::stream_ref stream)
  {
    this->merge_async(other, stream);
    stream.sync();
  }

  __host__ constexpr void merge_async(bloom_filter_impl<Key, Extent, Scope, Policy> const& other,
                                      cuda::stream_ref stream)
  {
    CUCO_EXPECTS(this->block_extent() == other.block_extent(),
                 "mismatching num_blocks in merge_async");
    CUCO_CUDA_TRY(cub::DeviceTransform::Transform(
      cuda::std::tuple{this->data(), other.data()},
      this->data(),
      this->block_extent() * words_per_block,
      [] __device__(word_type a, word_type b) { return a | b; },
      stream.get()));
  }

  __host__ constexpr void intersect(bloom_filter_impl<Key, Extent, Scope, Policy> const& other,
                                    cuda::stream_ref stream)
  {
    this->intersect_async(other, stream);
    stream.sync();
  }

  __host__ constexpr void intersect_async(
    bloom_filter_impl<Key, Extent, Scope, Policy> const& other, cuda::stream_ref stream)
  {
    CUCO_EXPECTS(this->block_extent() == other.block_extent(),
                 "mismatching num_blocks in intersect_async");
    CUCO_CUDA_TRY(cub::DeviceTransform::Transform(
      cuda::std::tuple{this->data(), other.data()},
      this->data(),
      this->block_extent() * words_per_block,
      [] __device__(word_type a, word_type b) { return a & b; },
      stream.get()));
  }

  [[nodiscard]] __host__ __device__ constexpr word_type* data() noexcept { return words_; }

  [[nodiscard]] __host__ __device__ constexpr word_type const* data() const noexcept
  {
    return words_;
  }

  [[nodiscard]] __host__ __device__ constexpr extent_type block_extent() const noexcept
  {
    return num_blocks_;
  }

  template <bool ConditionalAdd = conditional_add, class BuildKey>
  __device__ void add(BuildKey build_key)
  {
    auto const [upper_hash, lower_hash] = policy_.split_hash(build_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);

    if constexpr (add_horizontal_layout == 1) {
      add_pattern<ConditionalAdd, 0>(block_index, lower_hash);
    } else {
#pragma unroll
      for (uint32_t thread_index = 0; thread_index < add_horizontal_layout; ++thread_index) {
        add_patterns<ConditionalAdd, 0>(block_index, lower_hash, thread_index);
      }
    }
  }

  template <bool ConditionalAdd = conditional_add, class CG, class BuildKey>
  __device__ void add(CG group, BuildKey build_key)
  {
    if constexpr (add_horizontal_layout == 1 || tile_size_v<CG> != add_horizontal_layout) {
      if (group.thread_rank() == 0) { this->template add<ConditionalAdd>(build_key); }
      group.sync();
    } else {
      auto const sh          = policy_.split_hash(build_key);
      auto const lower_hash  = sh.second;
      auto const block_index = policy_.block_index(sh.first, num_blocks_);

      add_patterns<ConditionalAdd, 0>(block_index, lower_hash, group.thread_rank());
    }
  }

  template <bool ConditionalAdd = conditional_add, class CG, class BuildKey>
  __device__ void add_coop(CG group, BuildKey build_key)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const [upper_hash, lower_hash] = policy_.split_hash(build_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);

#pragma unroll num_threads
    for (int i = 0; i < num_threads; ++i) {
      add_patterns<ConditionalAdd, 0>(
        group.shfl(block_index, i), group.shfl(lower_hash, i), group.thread_rank());
    }
  }

  template <bool ConditionalAdd = conditional_add, class CG, class InputIt, class Index>
  __device__ void add_coop(CG group, InputIt first, Index idx, bool is_valid)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    uint32_t upper_hash   = 0;
    uint32_t lower_hash   = 0;
    size_type block_index = 0;
    if (is_valid) {
      auto const& key = *(first + idx);
      auto const sh   = policy_.split_hash(key);
      upper_hash      = sh.first;
      lower_hash      = sh.second;
      block_index     = policy_.block_index(upper_hash, num_blocks_);
    }

#pragma unroll num_threads
    for (int i = 0; i < num_threads; ++i) {
      if (group.shfl(is_valid, i)) {
        add_patterns<ConditionalAdd, 0>(
          group.shfl(block_index, i), group.shfl(lower_hash, i), group.thread_rank());
      }
    }
  }

  template <class CG, class InputIt>
  __device__ void add(CG group, InputIt first, InputIt last)
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if constexpr (tile_size_v<CG> == add_horizontal_layout && add_horizontal_layout > 1) {
      auto constexpr num_threads = static_cast<cuco::detail::index_type>(tile_size_v<CG>);
      for (cuco::detail::index_type batch = 0; batch < num_keys; batch += num_threads) {
        auto const idx      = batch + static_cast<cuco::detail::index_type>(group.thread_rank());
        auto const is_valid = idx < num_keys;
        this->template add_coop<conditional_add>(group, first, idx, is_valid);
      }
    } else {
      auto const stride = static_cast<cuco::detail::index_type>(tile_size_v<CG>);
      for (cuco::detail::index_type i = static_cast<cuco::detail::index_type>(group.thread_rank());
           i < num_keys;
           i += stride) {
        this->add(*(first + i));
      }
    }
  }

  template <class InputIt>
  __host__ void add_async(InputIt first, InputIt last, cuda::stream_ref stream) noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = 256;
    auto constexpr cg_size    = static_cast<int32_t>(add_horizontal_layout);
    auto const grid_size      = cuco::detail::int_div_ceil(num_keys, block_size);

    detail::bloom_filter_ns::add_n<conditional_add, cg_size, block_size>
      <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, *this);
  }

  template <class InputIt>
  __host__ void add(InputIt first, InputIt last, cuda::stream_ref stream) noexcept
  {
    this->add_async(first, last, stream);
    stream.sync();
  }

  template <class ProbeKey>
  __device__ bool contains(ProbeKey probe_key) const
  {
    auto const [upper_hash, lower_hash] = policy_.split_hash(probe_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);

    if constexpr (contains_horizontal_layout == 1) {
      return compare_pattern<0>(block_index, lower_hash);
    } else {
      bool result = true;
#pragma unroll
      for (uint32_t thread_index = 0; thread_index < contains_horizontal_layout; ++thread_index) {
        result = result && compare_patterns<0>(block_index, lower_hash, thread_index);
      }
      return result;
    }
  }

  template <class CG, class ProbeKey>
  __device__ bool contains(CG group, ProbeKey probe_key) const
  {
    if constexpr (contains_horizontal_layout == 1 ||
                  tile_size_v<CG> != contains_horizontal_layout) {
      return this->contains(probe_key);
    } else {
      auto const sh          = policy_.split_hash(probe_key);
      auto const lower_hash  = sh.second;
      auto const block_index = policy_.block_index(sh.first, num_blocks_);

      return group.all(compare_patterns<0>(block_index, lower_hash, group.thread_rank()));
    }
  }

  template <class CG, class ProbeKey>
  __device__ bool contains_coop(CG group, ProbeKey probe_key) const
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const [upper_hash, lower_hash] = policy_.split_hash(probe_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);
    bool result_out                     = false;

#pragma unroll num_threads
    for (int i = 0; i < num_threads; ++i) {
      auto const result = group.all(compare_patterns<0>(
        group.shfl(block_index, i), group.shfl(lower_hash, i), group.thread_rank()));
      if (i == group.thread_rank()) { result_out = result; }
    }
    return result_out;
  }

  template <class CG, class InputIt, class Index>
  __device__ bool contains_coop(CG group, InputIt first, Index idx, bool is_valid) const
  {
    constexpr auto num_threads = tile_size_v<CG>;

    uint32_t upper_hash   = 0;
    uint32_t lower_hash   = 0;
    size_type block_index = 0;
    if (is_valid) {
      auto const& key = *(first + idx);
      auto const sh   = policy_.split_hash(key);
      upper_hash      = sh.first;
      lower_hash      = sh.second;
      block_index     = policy_.block_index(upper_hash, num_blocks_);
    }

    bool result_out = false;
#pragma unroll num_threads
    for (int i = 0; i < num_threads; ++i) {
      if (group.shfl(is_valid, i)) {
        auto const result = group.all(compare_patterns<0>(
          group.shfl(block_index, i), group.shfl(lower_hash, i), group.thread_rank()));
        if (i == group.thread_rank()) { result_out = result; }
      }
    }
    return result_out;
  }

  template <class CG, class InputIt, class OutputIt>
  __device__ void contains(CG group, InputIt first, InputIt last, OutputIt output_begin) const
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if constexpr (tile_size_v<CG> == contains_horizontal_layout && contains_horizontal_layout > 1) {
      auto constexpr num_threads = static_cast<cuco::detail::index_type>(tile_size_v<CG>);
      for (cuco::detail::index_type batch = 0; batch < num_keys; batch += num_threads) {
        auto const idx      = batch + static_cast<cuco::detail::index_type>(group.thread_rank());
        auto const is_valid = idx < num_keys;
        auto const result   = this->contains_coop(group, first, idx, is_valid);
        if (is_valid) { *(output_begin + idx) = result; }
      }
    } else {
      auto const stride = static_cast<cuco::detail::index_type>(tile_size_v<CG>);
      for (cuco::detail::index_type i = static_cast<cuco::detail::index_type>(group.thread_rank());
           i < num_keys;
           i += stride) {
        *(output_begin + i) = this->contains(*(first + i));
      }
    }
  }

  template <class InputIt, class OutputIt>
  __host__ void contains_async(InputIt first,
                               InputIt last,
                               OutputIt output_begin,
                               cuda::stream_ref stream) const noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = 256;
    auto constexpr cg_size    = static_cast<int32_t>(contains_horizontal_layout);
    auto const grid_size      = cuco::detail::int_div_ceil(num_keys, block_size);

    detail::bloom_filter_ns::contains_n<cg_size, block_size>
      <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, output_begin, *this);
  }

  template <class InputIt, class OutputIt>
  __host__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         cuda::stream_ref stream) const noexcept
  {
    this->contains_async(first, last, output_begin, stream);
    stream.sync();
  }

  template <class InputIt, class StencilIt, class Predicate>
  __host__ void add_if_async(InputIt first,
                             InputIt last,
                             StencilIt stencil,
                             Predicate pred,
                             cuda::stream_ref stream) noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = 256;
    auto constexpr cg_size    = static_cast<int32_t>(add_horizontal_layout);
    auto const grid_size      = cuco::detail::int_div_ceil(num_keys, block_size);

    detail::bloom_filter_ns::add_if_n<conditional_add, cg_size, block_size>
      <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, stencil, pred, *this);
  }

  template <class InputIt, class StencilIt, class Predicate>
  __host__ void add_if(InputIt first,
                       InputIt last,
                       StencilIt stencil,
                       Predicate pred,
                       cuda::stream_ref stream) noexcept
  {
    this->add_if_async(first, last, stencil, pred, stream);
    stream.sync();
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void contains_if_async(InputIt first,
                                  InputIt last,
                                  StencilIt stencil,
                                  Predicate pred,
                                  OutputIt output_begin,
                                  cuda::stream_ref stream) const noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = 256;
    auto constexpr cg_size    = static_cast<int32_t>(contains_horizontal_layout);
    auto const grid_size      = cuco::detail::int_div_ceil(num_keys, block_size);

    detail::bloom_filter_ns::contains_if_n<cg_size, block_size>
      <<<grid_size, block_size, 0, stream.get()>>>(
        first, num_keys, stencil, pred, output_begin, *this);
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void contains_if(InputIt first,
                            InputIt last,
                            StencilIt stencil,
                            Predicate pred,
                            OutputIt output_begin,
                            cuda::stream_ref stream) const noexcept
  {
    this->contains_if_async(first, last, stencil, pred, output_begin, stream);
    stream.sync();
  }

  //  private:
  template <uint32_t NumWords>
  __device__ constexpr cuda::std::array<word_type, NumWords> vec_load_words(size_type index) const
  {
    // The block storage is aligned to `alignment()`, but a per-lane load at offset `index` is
    // only guaranteed to be aligned to `min(NumWords * sizeof(word_type), alignment())`. Hand the
    // compiler the alignment that's actually delivered, not the block-level maximum.
    constexpr auto load_alignment =
      cuda::std::min<size_t>(NumWords * sizeof(word_type), alignment());
    return *reinterpret_cast<cuda::std::array<word_type, NumWords>*>(
      __builtin_assume_aligned(words_ + index, load_alignment));
  }

  template <bool ConditionalAdd, uint32_t LoopIndex>
  __device__ constexpr void add_pattern(uint32_t block_index, uint32_t lower_hash)
  {
    static_assert(add_horizontal_layout == 1, "add_pattern() requires add_horizontal_layout == 1");

    if constexpr (LoopIndex < add_loop_count) {
      auto const pattern =
        policy_.template array_pattern<LoopIndex, add_vertical_layout>(lower_hash);
      auto* word_base = words_ + block_index * words_per_block + LoopIndex * add_vertical_layout;

      for (int i = 0; i < add_vertical_layout; ++i) {
        atomic_or<ConditionalAdd>(word_base + i, pattern[i]);
      }

      // Recurse.
      add_pattern<ConditionalAdd, LoopIndex + 1>(block_index, lower_hash);
    }
  }

  template <bool ConditionalAdd, uint32_t LoopIndex>
  __device__ constexpr void add_patterns(uint32_t block_index,
                                         uint32_t lower_hash,
                                         uint32_t thread_index)
  {
    static_assert(add_horizontal_layout > 1, "add_patterns() requires add_horizontal_layout > 1");

    if constexpr (LoopIndex < add_loop_count) {
      auto const pattern =
        policy_.template array_pattern<LoopIndex, add_horizontal_layout, add_vertical_layout>(
          lower_hash, thread_index);
      auto* word_base = words_ + block_index * words_per_block +
                        LoopIndex * add_vertical_layout * add_horizontal_layout +
                        thread_index * add_vertical_layout;

      for (int i = 0; i < add_vertical_layout; ++i) {
        atomic_or<ConditionalAdd>(word_base + i, pattern[i]);
      }

      // Recurse.
      add_patterns<ConditionalAdd, LoopIndex + 1>(block_index, lower_hash, thread_index);
    }
  }

  template <bool ConditionalAdd>
  __device__ constexpr void atomic_or(word_type* word_ptr, word_type pattern) const
  {
    // Native atomicOr: cuda::atomic_ref::fetch_or produces consistently slower codegen here.
    auto const do_or = [&]() {
      auto* const p = reinterpret_cast<atomic_word_type*>(word_ptr);
      auto const v  = static_cast<atomic_word_type>(pattern);
      if constexpr (thread_scope == cuda::thread_scope_thread) {
        *p |= v;
      } else if constexpr (thread_scope == cuda::thread_scope_block) {
        atomicOr_block(p, v);
      } else if constexpr (thread_scope == cuda::thread_scope_device) {
        atomicOr(p, v);
      } else if constexpr (thread_scope == cuda::thread_scope_system) {
        atomicOr_system(p, v);
      } else {
        static_assert(cuco::dependent_false<word_type>,
                      "unsupported cuda::thread_scope for native atomic_or");
      }
    };

    if constexpr (ConditionalAdd) {
      // Benign non-atomic read racing with atomicOr; technically UB but used throughout cuco.
      if ((*word_ptr & pattern) != pattern) { do_or(); }
    } else {
      do_or();
    }
  }

  /// Compare the stored pattern against the expected pattern for the given hash value.
  // Precondition: contains_horizontal_layout == 1
  template <uint32_t LoopIndex>
  __device__ constexpr bool compare_pattern(uint32_t block_index, uint32_t lower_hash) const
  {
    static_assert(contains_horizontal_layout == 1,
                  "compare_pattern() requires contains_horizontal_layout == 1");

    if constexpr (LoopIndex < contains_loop_count) {
      auto const stored_pattern = this->vec_load_words<contains_vertical_layout>(
        block_index * words_per_block + LoopIndex * contains_vertical_layout);
      auto const expected_pattern =
        policy_.template array_pattern<LoopIndex, contains_vertical_layout>(lower_hash);

      bool match = true;
      for (int i = 0; i < contains_vertical_layout; ++i) {
        match &= (stored_pattern[i] & expected_pattern[i]) == expected_pattern[i];
      }

      // Recurse.
      // Early exit in this implementation occurs at the granulairy of contains_vertical_layout
      // words.
      if constexpr (early_exit_contains) {
        if (!match) { return false; }
        return compare_pattern<LoopIndex + 1>(block_index, lower_hash);
      } else {
        return compare_pattern<LoopIndex + 1>(block_index, lower_hash) && match;
      }
    } else {
      return true;
    }
  }

  template <uint32_t LoopIndex>
  __device__ constexpr bool compare_patterns(uint32_t block_index,
                                             uint32_t lower_hash,
                                             uint32_t thread_index) const
  {
    static_assert(contains_horizontal_layout > 1,
                  "compare_patterns() requires HorizontalLayout > 1");

    if constexpr (LoopIndex < contains_loop_count) {
      auto const stored_pattern = this->vec_load_words<contains_vertical_layout>(
        block_index * words_per_block +
        LoopIndex * contains_vertical_layout * contains_horizontal_layout +
        thread_index * contains_vertical_layout);
      auto const expected_pattern =
        policy_
          .template array_pattern<LoopIndex, contains_horizontal_layout, contains_vertical_layout>(
            lower_hash, thread_index);

      bool match = true;
      for (int i = 0; i < contains_vertical_layout; ++i) {
        match &= (stored_pattern[i] & expected_pattern[i]) == expected_pattern[i];
      }

      // Per-thread early exit: short-circuit this thread's recursion if its slice already missed.
      if constexpr (early_exit_contains) {
        if (!match) { return false; }
        return compare_patterns<LoopIndex + 1>(block_index, lower_hash, thread_index);
      } else {
        return compare_patterns<LoopIndex + 1>(block_index, lower_hash, thread_index) && match;
      }
    } else {
      return true;
    }
  }

  word_type* words_;
  extent_type num_blocks_;
  policy_type policy_;
};

}  // namespace cuco::detail
