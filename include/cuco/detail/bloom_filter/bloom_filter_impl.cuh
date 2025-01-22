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

#include <cuco/detail/bloom_filter/kernels.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/utility/cuda.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/detail/utils.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cub/device/device_for.cuh>
#include <cuda/atomic>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>  // TODO #include <cuda/std/algorithm> once available
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/stream_ref>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

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

  static constexpr auto thread_scope    = Scope;
  static constexpr auto words_per_block = policy_type::words_per_block;

 private:
  __host__ __device__ static constexpr size_t max_vec_bytes() noexcept
  {
    constexpr auto word_bytes  = sizeof(word_type);
    constexpr auto block_bytes = word_bytes * words_per_block;
    return cuda::std::min(cuda::std::max(word_bytes, 32ul),
                          block_bytes);  // aiming for 2xLDG128 -> 1 sector per thread
  }

 public:
  struct alignas(max_vec_bytes()) filter_block_type {
   private:
    word_type data_[words_per_block];
  };

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

  __host__ __device__ explicit constexpr bloom_filter_impl(filter_block_type* filter,
                                                           Extent num_blocks,
                                                           cuda_thread_scope<Scope>,
                                                           Policy policy) noexcept
    : words_{reinterpret_cast<word_type*>(filter)}, num_blocks_{num_blocks}, policy_{policy}
  {
  }

  __host__ __device__ explicit constexpr bloom_filter_impl(word_type* filter,
                                                           Extent num_blocks,
                                                           cuda_thread_scope<Scope>,
                                                           Policy policy) noexcept
    : words_{filter}, num_blocks_{num_blocks}, policy_{policy}
  {
  }

  template <class CG>
  __device__ constexpr void clear(CG const& group)
  {
    for (int i = group.thread_rank(); num_blocks_ * words_per_block; i += group.size()) {
      words_[i] = 0;
    }
  }

  __host__ constexpr void clear(cuda::stream_ref stream)
  {
    this->clear_async(stream);
    stream.wait();
  }

  __host__ constexpr void clear_async(cuda::stream_ref stream)
  {
    CUCO_CUDA_TRY(cub::DeviceFor::ForEachN(
      words_,
      num_blocks_ * words_per_block,
      [] __device__(word_type & word) { word = 0; },
      stream.get()));
  }

  template <class ProbeKey>
  __device__ void add(ProbeKey const& key)
  {
    auto const hash_value = policy_.hash(key);
    auto const idx        = policy_.block_index(hash_value, num_blocks_);

#pragma unroll words_per_block
    for (uint32_t i = 0; i < words_per_block; ++i) {
      auto const word = policy_.word_pattern(hash_value, i);
      if (word != 0) {
        auto atom_word =
          cuda::atomic_ref<word_type, thread_scope>{*(words_ + (idx * words_per_block + i))};
        atom_word.fetch_or(word, cuda::memory_order_relaxed);
      }
    }
  }

  template <class CG, class ProbeKey>
  __device__ void add(CG const& group, ProbeKey const& key)
  {
    constexpr auto num_threads         = tile_size_v<CG>;
    constexpr auto optimal_num_threads = add_optimal_cg_size();
    constexpr auto words_per_thread    = words_per_block / optimal_num_threads;

    // If single thread is optimal, use scalar add
    if constexpr (num_threads == 1 or optimal_num_threads == 1) {
      this->add(key);
    } else {
      auto const rank = group.thread_rank();

      auto const hash_value = policy_.hash(key);
      auto const idx        = policy_.block_index(hash_value, num_blocks_);

#pragma unroll
      for (uint32_t i = rank; i < optimal_num_threads; i += num_threads) {
        auto const word = policy_.word_pattern(hash_value, rank);

        auto atom_word =
          cuda::atomic_ref<word_type, thread_scope>{*(words_ + (idx * words_per_block + rank))};
        atom_word.fetch_or(word, cuda::memory_order_relaxed);
      }
    }
  }

  template <class InputIt>
  __host__ constexpr void add(InputIt first, InputIt last, cuda::stream_ref stream)
  {
    this->add_async(first, last, stream);
    stream.wait();
  }

  template <class InputIt>
  __host__ constexpr void add_async(InputIt first, InputIt last, cuda::stream_ref stream)
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    if constexpr (words_per_block == 1) {
      CUCO_CUDA_TRY(cub::DeviceFor::ForEachCopyN(
        first,
        num_keys,
        [*this] __device__(key_type const key) mutable { this->add(key); },
        stream.get()));
    } else {
      auto const always_true = thrust::constant_iterator<bool>{true};
      this->add_if_async(first, last, always_true, thrust::identity{}, stream);
    }
  }

  template <class InputIt, class StencilIt, class Predicate>
  __host__ constexpr void add_if(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream)
  {
    this->add_if_async(first, last, stencil, pred, stream);
    stream.wait();
  }

  template <class InputIt, class StencilIt, class Predicate>
  __host__ constexpr void add_if_async(InputIt first,
                                       InputIt last,
                                       StencilIt stencil,
                                       Predicate pred,
                                       cuda::stream_ref stream) noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr cg_size    = add_optimal_cg_size();
    auto constexpr block_size = cuco::detail::default_block_size();
    auto const grid_size =
      cuco::detail::grid_size(num_keys, cg_size, cuco::detail::default_stride(), block_size);

    detail::bloom_filter_ns::add_if_n<cg_size, block_size>
      <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, stencil, pred, *this);
  }

  template <class ProbeKey>
  [[nodiscard]] __device__ bool contains(ProbeKey const& key) const
  {
    auto const hash_value = policy_.hash(key);

    auto const stored_pattern = this->vec_load_words<words_per_block>(
      policy_.block_index(hash_value, num_blocks_) * words_per_block);

#pragma unroll words_per_block
    for (uint32_t i = 0; i < words_per_block; ++i) {
      auto const expected_pattern = policy_.word_pattern(hash_value, i);
      if ((stored_pattern[i] & expected_pattern) != expected_pattern) { return false; }
    }

    return true;
  }

  template <class CG, class ProbeKey>
  [[nodiscard]] __device__ bool contains(CG const& group, ProbeKey const& key) const
  {
    constexpr auto num_threads         = tile_size_v<CG>;
    constexpr auto optimal_num_threads = contains_optimal_cg_size();
    constexpr auto words_per_thread    = words_per_block / optimal_num_threads;

    // If single thread is optimal, use scalar contains
    if constexpr (num_threads == 1 or optimal_num_threads == 1) {
      return this->contains(key);
    } else {
      auto const rank       = group.thread_rank();
      auto const hash_value = policy_.hash(key);
      bool success          = true;

#pragma unroll
      for (uint32_t i = rank; i < optimal_num_threads; i += num_threads) {
        auto const thread_offset  = i * words_per_thread;
        auto const stored_pattern = this->vec_load_words<words_per_thread>(
          policy_.block_index(hash_value, num_blocks_) * words_per_block + thread_offset);
#pragma unroll words_per_thread
        for (uint32_t j = 0; j < words_per_thread; ++j) {
          auto const expected_pattern = policy_.word_pattern(hash_value, thread_offset + j);
          if ((stored_pattern[j] & expected_pattern) != expected_pattern) { success = false; }
        }
      }

      return group.all(success);
    }
  }

  // TODO
  // template <class CG, class InputIt, class OutputIt>
  // __device__ void contains(CG const& group, InputIt first, InputIt last, OutputIt output_begin)
  // const;

  template <class InputIt, class OutputIt>
  __host__ constexpr void contains(InputIt first,
                                   InputIt last,
                                   OutputIt output_begin,
                                   cuda::stream_ref stream) const
  {
    this->contains_async(first, last, output_begin, stream);
    stream.wait();
  }

  template <class InputIt, class OutputIt>
  __host__ constexpr void contains_async(InputIt first,
                                         InputIt last,
                                         OutputIt output_begin,
                                         cuda::stream_ref stream) const noexcept
  {
    auto const always_true = thrust::constant_iterator<bool>{true};
    this->contains_if_async(first, last, always_true, thrust::identity{}, output_begin, stream);
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ constexpr void contains_if(InputIt first,
                                      InputIt last,
                                      StencilIt stencil,
                                      Predicate pred,
                                      OutputIt output_begin,
                                      cuda::stream_ref stream) const
  {
    this->contains_if_async(first, last, stencil, pred, output_begin, stream);
    stream.wait();
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ constexpr void contains_if_async(InputIt first,
                                            InputIt last,
                                            StencilIt stencil,
                                            Predicate pred,
                                            OutputIt output_begin,
                                            cuda::stream_ref stream) const noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr cg_size    = contains_optimal_cg_size();
    auto constexpr block_size = cuco::detail::default_block_size();
    auto const grid_size =
      cuco::detail::grid_size(num_keys, cg_size, cuco::detail::default_stride(), block_size);

    detail::bloom_filter_ns::contains_if_n<cg_size, block_size>
      <<<grid_size, block_size, 0, stream.get()>>>(
        first, num_keys, stencil, pred, output_begin, *this);
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

  // TODO
  // [[nodiscard]] __host__ double occupancy() const;
  // [[nodiscard]] __host__ double expected_false_positive_rate(size_t unique_keys) const
  // [[nodiscard]] __host__ __device__ static uint32_t optimal_pattern_bits(size_t num_blocks)
  // template <typename CG, cuda::thread_scope NewScope = thread_scope>
  // [[nodiscard]] __device__ constexpr auto make_copy(CG const& group, word_type* const
  // memory_to_use, cuda_thread_scope<NewScope> scope = {}) const noexcept;

 private:
  template <uint32_t NumWords>
  __device__ constexpr cuda::std::array<word_type, NumWords> vec_load_words(size_type index) const
  {
    return *reinterpret_cast<cuda::std::array<word_type, NumWords>*>(__builtin_assume_aligned(
      words_ + index, cuda::std::min(sizeof(word_type) * NumWords, max_vec_bytes())));
  }

  [[nodiscard]] __host__ __device__ static constexpr int32_t add_optimal_cg_size()
  {
    return words_per_block;  // one thread per word so atomic updates can be coalesced
  }

  [[nodiscard]] __host__ __device__ static constexpr int32_t contains_optimal_cg_size()
  {
    constexpr auto word_bytes  = sizeof(word_type);
    constexpr auto block_bytes = word_bytes * words_per_block;
    return block_bytes / max_vec_bytes();  // one vector load per thread
  }

  word_type* words_;
  extent_type num_blocks_;
  policy_type policy_;
};

}  // namespace cuco::detail
