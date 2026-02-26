/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <cuco/detail/utility/math.cuh>
#include <cuco/detail/utils.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/atomic>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>  // TODO #include <cuda/std/algorithm> once available
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/stream_ref>
#include <cuda/utility>
#include <thrust/iterator/constant_iterator.h>

#include <cooperative_groups.h>

#include <cstdint>

namespace cuco::detail {

/**
 * @brief Device functor for adding a single key to the bloom filter
 *
 * This functor is used with cuda::static_for to iterate over all words in a filter block
 * and set the appropriate bits for a given key's hash value. Each iteration processes
 * one word in the block using atomic operations to ensure thread safety.
 *
 * @tparam HashValue Type of the hash value (typically uint64_t)
 * @tparam BlockIndex Type of the block index (typically size_t or uint32_t)
 * @tparam Policy Filter policy type that provides word pattern generation
 * @tparam WordType Underlying word type of the filter (typically uint64_t)
 * @tparam Scope CUDA thread scope for atomic operations
 */
template <typename HashValue,
          typename BlockIndex,
          typename Policy,
          typename WordType,
          cuda::thread_scope Scope>
struct add_impl_functor {
  HashValue hash_value;    ///< Hash value of the key being added
  BlockIndex block_index;  ///< Index of the filter block to modify
  Policy policy_;          ///< Filter policy for generating bit patterns
  WordType* words_;        ///< Pointer to the filter's word array
  size_t words_per_block;  ///< Number of words in each filter block

  /**
   * @brief Processes one word in the filter block for key insertion
   *
   * @tparam I Type of the integral constant passed by cuda::static_for
   * @param i Integral constant representing the word index within the block
   */
  template <typename I>
  __device__ void operator()(I i) const
  {
    auto const word = policy_.word_pattern(hash_value, i());
    if (word != 0) {
      auto atom_word =
        cuda::atomic_ref<WordType, Scope>{*(words_ + (block_index * words_per_block + i()))};
      atom_word.fetch_or(word, cuda::memory_order_relaxed);
    }
  }
};

/**
 * @brief Device functor for cooperative group-based batch key insertion
 *
 * This functor is used with cuda::static_for to process multiple keys in parallel
 * within a cooperative group. Each thread in the group processes a different key
 * using shuffle operations to share hash values and block indices across threads.
 *
 * @tparam CG Cooperative group type (e.g., thread_block_tile)
 * @tparam HashValue Type of the hash value
 * @tparam BlockIndex Type of the block index
 * @tparam BloomFilterImpl Type of the bloom filter implementation
 */
template <typename CG, typename HashValue, typename BlockIndex, typename BloomFilterImpl>
struct add_group_functor {
  CG group;                ///< Cooperative group for parallel processing
  HashValue hash_value;    ///< Hash value of the current thread's key
  BlockIndex block_index;  ///< Block index of the current thread's key
  size_t i;                ///< Starting index in the key batch
  size_t num_keys;         ///< Total number of keys to process
  size_t num_threads;      ///< Number of threads in the group
  BloomFilterImpl* self;   ///< Pointer to the bloom filter implementation

  /**
   * @brief Processes one thread's key in the cooperative group batch insertion
   *
   * @tparam J Type of the integral constant passed by cuda::static_for
   * @param j Integral constant representing the thread index within the group
   */
  template <typename J>
  __device__ void operator()(J j) const
  {
    if ((j() < num_threads) and (i + j() < num_keys)) {
      self->add_impl(group, group.shfl(hash_value, j()), group.shfl(block_index, j()));
    }
  }
};

/**
 * @brief Device functor for worker group-based batch key insertion
 *
 * This functor is used with cuda::static_for to process multiple keys in parallel
 * within a worker group (subdivision of a larger cooperative group). Similar to
 * add_group_functor but operates on a smaller worker group with offset handling
 * for processing different portions of the key batch.
 *
 * @tparam WorkerGroup Worker group type (subdivision of cooperative group)
 * @tparam HashValue Type of the hash value
 * @tparam BlockIndex Type of the block index
 * @tparam BloomFilterImpl Type of the bloom filter implementation
 */
template <typename WorkerGroup, typename HashValue, typename BlockIndex, typename BloomFilterImpl>
struct add_worker_group_functor {
  WorkerGroup worker_group;   ///< Worker group (subdivision of cooperative group)
  HashValue hash_value;       ///< Hash value of the current thread's key
  BlockIndex block_index;     ///< Block index of the current thread's key
  size_t i;                   ///< Starting index in the key batch
  size_t worker_offset;       ///< Offset for this worker group within the batch
  size_t num_keys;            ///< Total number of keys to process
  size_t worker_num_threads;  ///< Number of threads in the worker group
  BloomFilterImpl* self;      ///< Pointer to the bloom filter implementation

  /**
   * @brief Processes one thread's key in the worker group batch insertion
   *
   * @tparam J Type of the integral constant passed by cuda::static_for
   * @param j Integral constant representing the thread index within the worker group
   */
  template <typename J>
  __device__ void operator()(J j) const
  {
    if ((j() < worker_num_threads) and (i + worker_offset + j() < num_keys)) {
      self->add_impl(
        worker_group, worker_group.shfl(hash_value, j()), worker_group.shfl(block_index, j()));
    }
  }
};

/**
 * @brief Device functor for cooperative group-based single key insertion
 *
 * This functor is used with cuda::static_for to add a single key to the bloom filter
 * using a cooperative group. Each thread in the group processes different words in
 * the filter block based on thread rank and stride pattern. Used when the group size
 * doesn't match the number of words per block.
 *
 * @tparam HashValue Type of the hash value
 * @tparam BlockIndex Type of the block index
 * @tparam WordType Underlying word type of the filter
 * @tparam Scope CUDA thread scope for atomic operations
 * @tparam Policy Filter policy type that provides word pattern generation
 */
template <typename HashValue,
          typename BlockIndex,
          typename WordType,
          cuda::thread_scope Scope,
          typename Policy>
struct add_impl_group_functor {
  HashValue hash_value;    ///< Hash value of the key being added
  BlockIndex block_index;  ///< Index of the filter block to modify
  WordType* words_;        ///< Pointer to the filter's word array
  size_t words_per_block;  ///< Number of words in each filter block
  size_t rank;             ///< Thread rank within the cooperative group
  size_t num_threads;      ///< Number of threads in the cooperative group
  Policy policy_;          ///< Filter policy for generating bit patterns

  /**
   * @brief Processes one word in the filter block using cooperative group stride pattern
   *
   * @tparam I Type of the integral constant passed by cuda::static_for
   * @param i Integral constant representing the word index within the block
   */
  template <typename I>
  __device__ void operator()(I i) const
  {
    if (i() >= rank && (i() - rank) % num_threads == 0) {
      auto atom_word =
        cuda::atomic_ref<WordType, Scope>{*(words_ + (block_index * words_per_block + i()))};
      atom_word.fetch_or(policy_.word_pattern(hash_value, i()), cuda::memory_order_relaxed);
    }
  }
};

/**
 * @brief Device functor for checking if a key exists in the bloom filter
 *
 * This functor is used with cuda::static_for to iterate over all words in a filter block
 * and check if the expected bit patterns for a given key's hash value are present.
 * If any expected bit is missing, the result is set to false, indicating the key
 * is definitely not in the set.
 *
 * @tparam HashValue Type of the hash value
 * @tparam StoredPattern Type of the stored pattern array (typically array of WordType)
 * @tparam Policy Filter policy type that provides word pattern generation
 */
template <typename HashValue, typename StoredPattern, typename Policy>
struct contains_functor {
  HashValue hash_value;          ///< Hash value of the key being queried
  StoredPattern stored_pattern;  ///< Array of stored bit patterns from the filter block
  Policy policy_;                ///< Filter policy for generating expected bit patterns
  bool* result;                  ///< Pointer to result flag (set to false if key not found)

  /**
   * @brief Checks one word in the filter block for the expected bit pattern
   *
   * @tparam I Type of the integral constant passed by cuda::static_for
   * @param i Integral constant representing the word index within the block
   */
  template <typename I>
  __device__ void operator()(I i) const
  {
    auto const expected_pattern = policy_.word_pattern(hash_value, i());
    if ((stored_pattern[i()] & expected_pattern) != expected_pattern) { *result = false; }
  }
};

template <class Key, class Extent, cuda::thread_scope Scope, class Policy>
class bloom_filter_impl {
 public:
  using key_type    = Key;
  using extent_type = Extent;
  using size_type   = typename extent_type::value_type;
  using policy_type = Policy;
  using word_type   = typename policy_type::word_type;
  // uint64_t may be unsigned long, but atomicOr requires unsigned long long
  using atomic_word_type = typename cuda::std::
    conditional_t<cuda::std::is_same_v<word_type, unsigned long>, unsigned long long, word_type>;

  // These knobs need to be public for exposure to the kernel definitions
  // TODO remove these once we settled on a setup which works best
  static constexpr bool use_invoke_one                       = true;
  static constexpr bool use_early_exit                       = false;
  static constexpr bool use_cub_kernels                      = true;
  static constexpr bool use_warp_cooperative_add_kernel      = true;
  static constexpr bool use_warp_cooperative_contains_kernel = true;
  static constexpr bool use_work_stealing_add_kernel         = false;  // DEBUG
  static constexpr bool use_work_stealing_contains_kernel    = false;  // DEBUG
  static constexpr bool use_cuda_atomic_ref                  = false;

  static constexpr auto thread_scope    = Scope;
  static constexpr auto words_per_block = policy_type::words_per_block;

  static constexpr auto add_vertical_layout        = policy_type::add_vertical_layout;
  static constexpr auto add_horizontal_layout      = policy_type::add_horizontal_layout;
  static constexpr auto contains_vertical_layout   = policy_type::contains_vertical_layout;
  static constexpr auto contains_horizontal_layout = policy_type::contains_horizontal_layout;
  static constexpr auto add_loop_count =
    words_per_block / (add_vertical_layout * add_horizontal_layout);
  static constexpr auto contains_loop_count =
    words_per_block / (contains_vertical_layout * contains_horizontal_layout);

  //===----------Cache-Sectorized----------===//
  /// TODO: the following will break the compiler with non-parametric policies
  static constexpr bool is_cache_sectorized = policy_type::is_cache_sectorized;

  // TODO static_assert layout, word type, etc.
  static_assert((not use_cuda_atomic_ref) or (Scope == cuda::thread_scope::thread_scope_device),
                "atomicOr requires device scope");
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
  __device__ constexpr void clear(CG group)
  {
    // TODO optimize this
    for (int i = group.thread_rank(); i < static_cast<size_type>(num_blocks_) * words_per_block;
         i += group.size()) {
      words_[i] = 0;
    }
  }

  __host__ constexpr void clear(cuda::stream_ref stream)
  {
    this->clear_async(stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  __host__ constexpr void clear_async(cuda::stream_ref stream) noexcept
  {
    cub::DeviceFor::ForEachN(
      words_,
      static_cast<size_type>(num_blocks_) * words_per_block,
      [] __device__(word_type & word) { word = 0; },
      stream.get());
  }

  template <class HashValue, class BlockIndex>
  __device__ void add_impl(HashValue const& hash_value, BlockIndex block_index)
  {
    add_impl_functor<HashValue, BlockIndex, policy_type, word_type, thread_scope> functor{
      hash_value, block_index, policy_, words_, words_per_block};
    cuda::static_for<words_per_block>(functor);
  }

  template <class CG, class HashValue, class BlockIndex>
  __device__ void add_impl(CG group, HashValue hash_value, BlockIndex block_index)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const block_offset = block_index * words_per_block;

    if constexpr (num_threads == 1) {
      this->add_impl(hash_value, block_index);
    } else if constexpr (num_threads == (words_per_block / add_vertical_layout)) {
      auto const thread_offset = group.thread_rank() * add_vertical_layout;
#pragma unroll add_vertical_layout
      for (uint32_t j = 0; j < add_vertical_layout; ++j) {
        auto const word = policy_.word_pattern(hash_value, thread_offset + j);
        auto atom_word =
          cuda::atomic_ref<word_type, thread_scope>{*(words_ + block_offset + thread_offset + j)};
        atom_word.fetch_or(word, cuda::memory_order_relaxed);
      }
    } else {
#pragma unroll
      for (uint32_t i = group.thread_rank(); i < words_per_block / add_vertical_layout;
           i += num_threads) {
        auto const thread_offset = i * add_vertical_layout;
#pragma unroll add_vertical_layout
        for (uint32_t j = 0; j < add_vertical_layout; ++j) {
          auto const word = policy_.word_pattern(hash_value, thread_offset + j);
          auto atom_word =
            cuda::atomic_ref<word_type, thread_scope>{*(words_ + block_offset + thread_offset + j)};
          atom_word.fetch_or(word, cuda::memory_order_relaxed);
        }
      }
    }
  }

  template <class ProbeKey>
  __device__ void add(ProbeKey const& key)
  {
    auto const hash_value = policy_.hash(key);
    this->add_impl(hash_value, policy_.block_index(hash_value, num_blocks_));
  }

  template <class InputIt>
  __device__ void add(InputIt first, InputIt last)
  {
    auto const num_keys = cuco::detail::distance(first, last);
    for (decltype(num_keys) i = 0; i < num_keys; ++i) {
      auto const hash_value = policy_.hash(*(first + i));
      this->add_impl(hash_value, policy_.block_index(hash_value, num_blocks_));
    }
  }

  template <class CG, class ProbeKey>
  __device__ void add(CG group, ProbeKey const& key)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    // If single thread is optimal, use scalar contains
    if constexpr (num_threads == 1) {
      this->add(key);
    } else {
      // TODO check if invoke_one this isn't hurting performance
      if constexpr (use_invoke_one) {
        auto const [hash_value, block_index] =
          cooperative_groups::invoke_one_broadcast(group, [*this, &key] __device__() {
            auto const hash_value = policy_.hash(key);
            return cuda::std::tuple{hash_value, policy_.block_index(hash_value, num_blocks_)};
          });
        this->add_impl(group, hash_value, block_index);
      } else {
        auto const hash_value  = policy_.hash(key);
        auto const block_index = policy_.block_index(hash_value, num_blocks_);
        this->add_impl(group, hash_value, block_index);
      }
    }
  }

  template <class CG, class InputIt>
  __device__ void add(CG group, InputIt first, InputIt last)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    if constexpr (num_threads == 1) {
      for (decltype(num_keys) i = 0; i < num_keys; ++i) {
        typename cuda::std::iterator_traits<InputIt>::value_type const& key{*(first + i)};
        this->add(key);
      }
    } else {
      typename policy_type::hash_result_type hash_value;
      size_type block_index;

      auto const rank        = group.thread_rank();
      auto const group_iters = cuco::detail::int_div_ceil(num_keys, num_threads);

      if constexpr (num_threads <= add_horizontal_layout) {
        for (cuda::std::remove_const_t<decltype(num_keys)> i = 0; i < group_iters; ++i) {
          auto const group_offset = i * num_threads;
          if (group_offset + rank < num_keys) {
            typename cuda::std::iterator_traits<InputIt>::value_type const& key{
              *(first + group_offset + rank)};
            hash_value  = policy_.hash(key);
            block_index = policy_.block_index(hash_value, num_blocks_);
          }

          for (uint32_t j = 0; (j < num_threads) and (group_offset + j < num_keys); ++j) {
            this->add_impl(group, group.shfl(hash_value, j), group.shfl(block_index, j));
          }
        }
      } else /* num_threads > horizontal_layout */ {
        // subdivide given CG into multiple optimal CGs
        auto const worker_group =
          cooperative_groups::tiled_partition<add_horizontal_layout, CG>(group);
        auto const worker_offset = add_horizontal_layout * worker_group.meta_group_rank();

        for (cuda::std::remove_const_t<decltype(num_keys)> i = 0; i < group_iters; ++i) {
          auto const group_offset = i * num_threads;
          if (group_offset + rank < num_keys) {
            typename cuda::std::iterator_traits<InputIt>::value_type const& key{
              *(first + group_offset + rank)};
            hash_value  = policy_.hash(key);
            block_index = policy_.block_index(hash_value, num_blocks_);
          }

          for (uint32_t j = 0;
               (j < add_horizontal_layout) and (group_offset + worker_offset + j < num_keys);
               ++j) {
            this->add_impl(
              worker_group, worker_group.shfl(hash_value, j), worker_group.shfl(block_index, j));
          }
        }
      }
    }
  }

  template <class InputIt>
  __host__ constexpr void add(InputIt first, InputIt last, cuda::stream_ref stream)
  {
    this->add_async(first, last, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  template <class InputIt>
  __host__ constexpr void add_async(InputIt first, InputIt last, cuda::stream_ref stream) noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    if constexpr (use_cub_kernels and ((words_per_block / add_vertical_layout) == 1)) {
      cub::DeviceFor::ForEachCopyN(
        first,
        num_keys,
        [*this] __device__(key_type const key) mutable { this->add(key); },
        stream.get());
    } else {
      auto const num_keys = cuco::detail::distance(first, last);
      if (num_keys == 0) { return; }

      auto constexpr block_size = cuco::detail::default_block_size();
      void const* kernel        = reinterpret_cast<void const*>(
        detail::bloom_filter_ns::add<block_size, InputIt, bloom_filter_impl>);
      auto const grid_size = cuco::detail::max_occupancy_grid_size(block_size, kernel);

      detail::bloom_filter_ns::add<block_size>
        <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, *this);
    }
  }

  template <class InputIt, class StencilIt, class Predicate>
  __host__ constexpr void add_if(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream)
  {
    this->add_if_async(first, last, stencil, pred, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
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

    auto constexpr cg_size    = add_horizontal_layout;
    auto constexpr block_size = cuco::detail::default_block_size();
    auto const grid_size =
      cuco::detail::grid_size(num_keys, cg_size, cuco::detail::default_stride(), block_size);

    detail::bloom_filter_ns::add_if_n<cg_size, block_size>
      <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, stencil, pred, *this);
  }

  template <class HashValue, class BlockIndex>
  __device__ bool contains_impl(HashValue hash_value, BlockIndex block_index) const
  {
    if constexpr (use_early_exit) {
#pragma unroll words_per_block / contains_vertical_layout
      for (uint32_t i = 0; i < words_per_block / contains_vertical_layout; ++i) {
        auto const stored_pattern = this->vec_load_words<contains_vertical_layout>(
          block_index * words_per_block + i * contains_vertical_layout);
#pragma unroll contains_vertical_layout
        for (uint32_t j = 0; j < contains_vertical_layout; ++j) {
          auto const word_offset      = i * contains_vertical_layout + j;
          auto const expected_pattern = policy_.word_pattern(hash_value, word_offset);
          // TODO we can replace this with a check against "is 0" if we negate the filter bits
          if ((stored_pattern[j] & expected_pattern) != expected_pattern) { return false; }
        }
      }
      return true;
    } else {
      /// NON-EARLY EXIT CODE PATH ///
      bool success = true;

#pragma unroll words_per_block / contains_vertical_layout
      for (uint32_t i = 0; i < words_per_block / contains_vertical_layout; ++i) {
        auto const stored_pattern = this->vec_load_words<contains_vertical_layout>(
          block_index * words_per_block + i * contains_vertical_layout);
#pragma unroll contains_vertical_layout
        for (uint32_t j = 0; j < contains_vertical_layout; ++j) {
          auto const word_offset      = i * contains_vertical_layout + j;
          auto const expected_pattern = policy_.word_pattern(hash_value, word_offset);
          success &= ((stored_pattern[j] & expected_pattern) == expected_pattern);
        }
      }
      return success;
    }
  }

  template <class CG, class HashValue, class BlockIndex>
  __device__ bool contains_impl(CG group, HashValue hash_value, BlockIndex block_index) const
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const block_offset = block_index * words_per_block;

    if constexpr (num_threads == 1) {
      return this->contains_impl(hash_value, block_index);
    } else {
      auto const rank = group.thread_rank();
      bool success    = true;
      if constexpr (use_early_exit) {
#pragma unroll
        for (uint32_t i = 0;
             i < int_div_ceil(words_per_block / contains_vertical_layout, num_threads);
             ++i) {
          auto const thread_offset = (i * num_threads + rank) * contains_vertical_layout;
          if (thread_offset < words_per_block) {
            auto const stored_pattern =
              this->vec_load_words<contains_vertical_layout>(block_offset + thread_offset);
#pragma unroll contains_vertical_layout
            for (uint32_t j = 0; j < contains_vertical_layout; ++j) {
              auto const expected_pattern = policy_.word_pattern(hash_value, thread_offset + j);
              // TODO we can replace this with a check against "is 0" if we negate the filter bits
              if ((stored_pattern[j] & expected_pattern) != expected_pattern) { success = false; }
            }
          }
          if (not group.all(success)) { return false; }  // early exit
        }
        return true;
      } else {
#pragma unroll
        for (uint32_t i = rank; i < words_per_block / contains_vertical_layout; i += num_threads) {
          auto const thread_offset = i * contains_vertical_layout;
          auto const stored_pattern =
            this->vec_load_words<contains_vertical_layout>(block_offset + thread_offset);
#pragma unroll contains_vertical_layout
          for (uint32_t j = 0; j < contains_vertical_layout; ++j) {
            auto const expected_pattern = policy_.word_pattern(hash_value, thread_offset + j);
            // TODO we can replace this with a check against "is 0" if we negate the filter bits
            if ((stored_pattern[j] & expected_pattern) != expected_pattern) { success = false; }
          }
        }
        return group.all(success);
      }
    }
  }

  template <class ProbeKey>
  [[nodiscard]] __device__ bool contains(ProbeKey const& key) const
  {
    auto const hash_value = policy_.hash(key);
    return this->contains_impl(hash_value, policy_.block_index(hash_value, num_blocks_));
  }

  template <class InputIt, class OutputIt>
  __device__ void contains(InputIt first, InputIt last, OutputIt output_begin) const
  {
    auto const num_keys = cuco::detail::distance(first, last);
    for (decltype(num_keys) i = 0; i < num_keys; ++i) {
      auto const hash_value = policy_.hash(*(first + i));
      *(output_begin + i) =
        this->contains_impl(hash_value, policy_.block_index(hash_value, num_blocks_));
    }
  }

  template <class CG, class ProbeKey>
  [[nodiscard]] __device__ bool contains(CG group, ProbeKey const& key) const
  {
    constexpr auto num_threads = tile_size_v<CG>;

    // If single thread is optimal, use scalar contains
    if constexpr (num_threads == 1) {
      return this->contains(key);
    } else {
      // TODO check if invoke_one this isn't hurting performance
      if constexpr (use_invoke_one) {
        auto const [hash_value, block_index] =
          cooperative_groups::invoke_one_broadcast(group, [*this, &key] __device__() {
            auto const hash_value = policy_.hash(key);
            return cuda::std::tuple{hash_value, policy_.block_index(hash_value, num_blocks_)};
          });
        return this->contains_impl(group, hash_value, block_index);
      } else {
        auto const hash_value  = policy_.hash(key);
        auto const block_index = policy_.block_index(hash_value, num_blocks_);
        return this->contains_impl(group, hash_value, block_index);
      }
    }
  }

  template <class CG, class InputIt, class OutputIt>
  __device__ void contains(CG group, InputIt first, InputIt last, OutputIt output_begin) const
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    if constexpr (num_threads == 1) {
      for (decltype(num_keys) i = 0; i < num_keys; ++i) {
        typename cuda::std::iterator_traits<InputIt>::value_type const& key{*(first + i)};
        *(output_begin + i) = this->contains(key);
      }
    } else {
      typename policy_type::hash_result_type hash_value;
      size_type block_index;
      bool result_out;

      auto const rank        = group.thread_rank();
      auto const group_iters = cuco::detail::int_div_ceil(num_keys, num_threads);

      if constexpr (num_threads <= contains_horizontal_layout) {
        for (cuda::std::remove_const_t<decltype(num_keys)> i = 0; i < group_iters; ++i) {
          auto const group_offset = i * num_threads;
          // coalesced input read
          if (group_offset + rank < num_keys) {
            typename cuda::std::iterator_traits<InputIt>::value_type const& key{
              *(first + group_offset + rank)};
            hash_value  = policy_.hash(key);
            block_index = policy_.block_index(hash_value, num_blocks_);
          }

          // group-wise cooperative lookup
          for (uint32_t j = 0; (j < num_threads) and (group_offset + j < num_keys); ++j) {
            bool result =
              this->contains_impl(group, group.shfl(hash_value, j), group.shfl(block_index, j));
            if (j == rank) { result_out = result; }
          }

          // coalesced output write
          if (group_offset + rank < num_keys) {
            *(output_begin + group_offset + rank) = result_out;
          }
        }
      } else /* num_threads > horizontal_layout */ {
        // subdivide given CG into multiple optimal CGs
        auto const worker_group =
          cooperative_groups::tiled_partition<contains_horizontal_layout, CG>(group);
        auto const worker_offset = contains_horizontal_layout * worker_group.meta_group_rank();
        auto const worker_rank   = worker_group.thread_rank();

        for (cuda::std::remove_const_t<decltype(num_keys)> i = 0; i < group_iters; ++i) {
          auto const group_offset = i * num_threads;
          // coalesced input read
          if (group_offset + rank < num_keys) {
            typename cuda::std::iterator_traits<InputIt>::value_type const& key{
              *(first + group_offset + rank)};
            hash_value  = policy_.hash(key);
            block_index = policy_.block_index(hash_value, num_blocks_);
          }

          // group-wise cooperative lookup
          for (uint32_t j = 0;
               (j < contains_horizontal_layout) and (group_offset + worker_offset + j < num_keys);
               ++j) {
            bool result = this->contains_impl(
              worker_group, worker_group.shfl(hash_value, j), worker_group.shfl(block_index, j));
            if (j == worker_rank) { result_out = result; }
          }

          // coalesced output write
          if (group_offset + rank < num_keys) {
            *(output_begin + group_offset + rank) = result_out;
          }
        }
      }
    }
  }

  template <class InputIt, class OutputIt>
  __host__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         cuda::stream_ref stream) const
  {
    this->contains_async(first, last, output_begin, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  template <class InputIt, class OutputIt>
  __host__ void contains_async(InputIt first,
                               InputIt last,
                               OutputIt output_begin,
                               cuda::stream_ref stream) const noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    if constexpr (use_cub_kernels and ((words_per_block / contains_vertical_layout) == 1)) {
      cub::DeviceTransform::Transform(
        first,
        output_begin,
        num_keys,
        [*this] __device__(auto const key) { return this->contains(key); },
        stream.get());
    } else {
      auto constexpr block_size = cuco::detail::default_block_size();
      auto constexpr cg_size    = static_cast<int32_t>(contains_horizontal_layout);
      auto const grid_size      = cuco::detail::int_div_ceil(num_keys * cg_size, block_size);

      detail::bloom_filter_ns::contains<block_size>
        <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, output_begin, *this);
    }
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void contains_if(InputIt first,
                            InputIt last,
                            StencilIt stencil,
                            Predicate pred,
                            OutputIt output_begin,
                            cuda::stream_ref stream) const
  {
    this->contains_if_async(first, last, stencil, pred, output_begin, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
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

    auto constexpr cg_size    = contains_horizontal_layout;
    auto constexpr block_size = cuco::detail::default_block_size();
    auto const grid_size =
      cuco::detail::grid_size(num_keys, cg_size, cuco::detail::default_stride(), block_size);

    detail::bloom_filter_ns::contains_if_n<cg_size, block_size>
      <<<grid_size, block_size, 0, stream.get()>>>(
        first, num_keys, stencil, pred, output_begin, *this);
  }

  __host__ constexpr void merge(bloom_filter_impl<Key, Extent, Scope, Policy> const& other,
                                cuda::stream_ref stream)
  {
    this->merge_async(other, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
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
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
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

  //===--------------------------------------------------===//
  // Parametric Filter Policy
  //===--------------------------------------------------===//
  // Kevin: Not implementing the <add/contains>_if() host-side entry points for now.

  // Single Thread Add
  template <bool ConditionalAtomic, class BuildKey>
  __device__ void add_exp(BuildKey build_key)
  {
    // Sanity checks. TODO: remove redundant checks.
    static_assert(add_horizontal_layout == 1, "This add_exp() requires add_horizontal_layout == 1");

    auto const [upper_hash, lower_hash] = policy_.split_hash(build_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);

    if constexpr (is_cache_sectorized) {
      auto const group_hash = lower_hash * policy_type::group_index_salt;
      add_pattern_cs<ConditionalAtomic, 0>(block_index, lower_hash, group_hash);
    } else {
      add_pattern<ConditionalAtomic, 0>(block_index, lower_hash);
    }
  }

  // Multi Thread Add
  template <bool ConditionalAtomic, class CG, class BuildKey>
  __device__ void add_exp(CG group, BuildKey build_key)
  {
    // Sanity checks. TODO: remove redundant checks.
    static_assert(add_horizontal_layout > 1, "This add_exp() requires add_horizontal_layout > 1");

    auto const [upper_hash, lower_hash] = policy_.split_hash(build_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);

    if constexpr (is_cache_sectorized) {
      auto const group_hash = lower_hash * policy_type::group_index_salt;
      add_patterns_cs<ConditionalAtomic, 0>(
        block_index, lower_hash, group_hash, group.thread_rank());
    } else {
      add_patterns<ConditionalAtomic, 0>(block_index, lower_hash, group.thread_rank());
    }
  }

  // Warp-cooperative Add
  template <bool ConditionalAtomic, class CG, class BuildKey>
  __device__ void add_exp_coop(CG group, BuildKey build_key)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const [upper_hash, lower_hash] = policy_.split_hash(build_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);

    if constexpr (is_cache_sectorized) {
      auto const group_hash = lower_hash * policy_type::group_index_salt;
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        add_patterns_cs<ConditionalAtomic, 0>(group.shfl(block_index, i),
                                              group.shfl(lower_hash, i),
                                              group.shfl(group_hash, i),
                                              group.thread_rank());
      }
    } else {
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        add_patterns<ConditionalAtomic, 0>(
          group.shfl(block_index, i), group.shfl(lower_hash, i), group.thread_rank());
      }
    }
  }

  template <bool ConditionalAtomic, class CG, class BuildKey>
  __device__ void add_exp_coop(CG group, BuildKey build_key, bool is_valid)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const [upper_hash, lower_hash] = policy_.split_hash(build_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);

    if constexpr (is_cache_sectorized) {
      auto const group_hash = lower_hash * policy_type::group_index_salt;
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        if (group.shfl(is_valid, i)) {
          add_patterns_cs<ConditionalAtomic, 0>(group.shfl(block_index, i),
                                                group.shfl(lower_hash, i),
                                                group.shfl(group_hash, i),
                                                group.thread_rank());
        }
      }
    } else {
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        if (group.shfl(is_valid, i)) {
          add_patterns<ConditionalAtomic, 0>(
            group.shfl(block_index, i), group.shfl(lower_hash, i), group.thread_rank());
        }
      }
    }
  }

  // Host-side Add Entry Points
  template <class InputIt>
  __host__ void add_exp_async(InputIt first, InputIt last, cuda::stream_ref stream) noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = 256;
    auto constexpr cg_size    = static_cast<int32_t>(add_horizontal_layout);
    auto const grid_size      = use_warp_cooperative_add_kernel
                                  ? cuco::detail::int_div_ceil(num_keys, block_size)
                                  : cuco::detail::int_div_ceil(num_keys * cg_size, block_size);
    auto const l2_cache_size  = static_cast<size_t>(cuco::detail::l2_cache_size());
    auto const filter_size    = static_cast<size_t>(static_cast<size_type>(num_blocks_)) *
                             words_per_block * sizeof(word_type);

    if (2 * filter_size < l2_cache_size) {
      if constexpr (use_work_stealing_add_kernel) {
        detail::bloom_filter_ns::add_exp_work_stealing_n<false, cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, *this);
      } else {
        detail::bloom_filter_ns::add_exp_n<false, cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, *this);
      }
    } else {
      if constexpr (use_work_stealing_add_kernel) {
        detail::bloom_filter_ns::add_exp_work_stealing_n<true, cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, *this);
      } else {
        detail::bloom_filter_ns::add_exp_n<true, cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, *this);
      }
    }
  }

  template <class InputIt>
  __host__ void add_exp(InputIt first, InputIt last, cuda::stream_ref stream) noexcept
  {
    this->add_exp_async(first, last, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  // Single Thread Contains
  template <class ProbeKey>
  __device__ bool contains_exp(ProbeKey probe_key) const
  {
    // Sanity checks. TODO: remove redundant checks.
    static_assert(contains_horizontal_layout == 1,
                  "This contains_exp() requires contains_horizontal_layout == 1");

    auto const [upper_hash, lower_hash] = policy_.split_hash(probe_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);

    if constexpr (is_cache_sectorized) {
      auto const group_hash = lower_hash * policy_type::group_index_salt;
      return compare_pattern_cs<0>(block_index, lower_hash, group_hash);
    } else {
      return compare_pattern<0>(block_index, lower_hash);
    }
  }

  // Multi Thread Contains
  template <class CG, class ProbeKey>
  __device__ bool contains_exp(CG group, ProbeKey probe_key) const
  {
    // Sanity checks. TODO: remove redundant checks.
    static_assert(contains_horizontal_layout > 1,
                  "This contains_exp() requires contains_horizontal_layout > 1");
    static_assert(tile_size_v<CG> == contains_horizontal_layout,
                  "This contains_exp() requires CG with size equal to contains_horizontal_layout");

    auto const [upper_hash, lower_hash] = policy_.split_hash(probe_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);

    if constexpr (is_cache_sectorized) {
      auto const group_hash = lower_hash * policy_type::group_index_salt;
      return compare_patterns_cs<0>(
        group, block_index, lower_hash, group_hash, group.thread_rank());
    } else {
      return compare_patterns<0>(group, block_index, lower_hash, group.thread_rank());
    }
  }

  // Warp-cooperative Contains
  template <class CG, class ProbeKey>
  __device__ bool contains_exp_coop(CG group, ProbeKey probe_key)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const [upper_hash, lower_hash] = policy_.split_hash(probe_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);
    bool result_out                     = false;

    if constexpr (is_cache_sectorized) {
      auto const group_hash = lower_hash * policy_type::group_index_salt;
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        auto const result = group.all(compare_patterns_cs<0>(group,
                                                             group.shfl(block_index, i),
                                                             group.shfl(lower_hash, i),
                                                             group.shfl(group_hash, i),
                                                             group.thread_rank()));
        if (i == group.thread_rank()) { result_out = result; }
      }
    } else {
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        auto const result = group.all(compare_patterns<0>(
          group, group.shfl(block_index, i), group.shfl(lower_hash, i), group.thread_rank()));
        if (i == group.thread_rank()) { result_out = result; }
      }
    }
    return result_out;
  }

  template <class CG, class ProbeKey>
  __device__ bool contains_exp_coop(CG group, ProbeKey probe_key, bool is_valid)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const [upper_hash, lower_hash] = policy_.split_hash(probe_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);
    bool result_out                     = false;

    if constexpr (is_cache_sectorized) {
      auto const group_hash = lower_hash * policy_type::group_index_salt;
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        if (group.shfl(is_valid, i)) {
          auto const result = group.all(compare_patterns_cs<0>(group,
                                                               group.shfl(block_index, i),
                                                               group.shfl(lower_hash, i),
                                                               group.shfl(group_hash, i),
                                                               group.thread_rank()));
          if (i == group.thread_rank()) { result_out = result; }
        }
      }
    } else {
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        if (group.shfl(is_valid, i)) {
          auto const result = group.all(compare_patterns<0>(
            group, group.shfl(block_index, i), group.shfl(lower_hash, i), group.thread_rank()));
          if (i == group.thread_rank()) { result_out = result; }
        }
      }
    }
    return result_out;
  }

  // Host-side Contains Entry Points
  template <class InputIt, class OutputIt>
  __host__ void contains_exp_async(InputIt first,
                                   InputIt last,
                                   OutputIt output_begin,
                                   cuda::stream_ref stream) const noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    if constexpr (use_cub_kernels and ((words_per_block / contains_vertical_layout) == 1)) {
      cub::DeviceTransform::Transform(
        first,
        output_begin,
        num_keys,
        [*this] __device__(auto const& key) { return this->contains_exp(key); },
        stream.get());
    } else {
      auto constexpr block_size = 256;
      auto constexpr cg_size    = static_cast<int32_t>(contains_horizontal_layout);
      auto const grid_size      = use_warp_cooperative_contains_kernel
                                    ? cuco::detail::int_div_ceil(num_keys, block_size)
                                    : cuco::detail::int_div_ceil(num_keys * cg_size, block_size);

      if constexpr (use_work_stealing_contains_kernel) {
        detail::bloom_filter_ns::contains_exp_work_stealing_n<cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, output_begin, *this);
      } else {
        detail::bloom_filter_ns::contains_exp_n<cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, output_begin, *this);
      }
    }
  }

  template <class InputIt, class OutputIt>
  __host__ void contains_exp(InputIt first,
                             InputIt last,
                             OutputIt output_begin,
                             cuda::stream_ref stream) const noexcept
  {
    this->contains_exp_async(first, last, output_begin, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  // TODO
  // [[nodiscard]] __host__ double occupancy() const;
  // [[nodiscard]] __host__ double expected_false_positive_rate(size_t unique_keys) const
  // [[nodiscard]] __host__ __device__ static uint32_t optimal_pattern_bits(size_t num_blocks)
  // template <typename CG, cuda::thread_scope NewScope = thread_scope>
  // [[nodiscard]] __device__ constexpr auto make_copy(CG group, word_type* const
  // memory_to_use, cuda_thread_scope<NewScope> scope = {}) const noexcept;

  //  private:
  template <uint32_t NumWords>
  __device__ constexpr cuda::std::array<word_type, NumWords> vec_load_words(size_type index) const
  {
    return *reinterpret_cast<cuda::std::array<word_type, NumWords>*>(
      __builtin_assume_aligned(words_ + index, alignment()));
  }

  //===--------------------------------------------------===//
  // Parametric Filter Policy
  //===--------------------------------------------------===//
  /// Insert the given pattern into the filter
  // Precondition: add_horizontal_layout == 1
  template <bool ConditionalAtomic, uint32_t LoopIndex>
  __device__ constexpr void add_pattern(uint32_t block_index, uint32_t lower_hash)
  {
    // Sanity check. TODO: remove redundant checks.
    static_assert(add_horizontal_layout == 1, "add_pattern() requires add_horizontal_layout == 1");

    if constexpr (LoopIndex < add_loop_count) {
      auto const pattern =
        policy_.template array_pattern<LoopIndex, add_vertical_layout>(lower_hash);
      auto* word_base = words_ + block_index * words_per_block + LoopIndex * add_vertical_layout;

      for (int i = 0; i < add_vertical_layout; ++i) {
        atomic_or<ConditionalAtomic>(word_base + i, pattern[i]);
      }

      // Recurse.
      add_pattern<ConditionalAtomic, LoopIndex + 1>(block_index, lower_hash);
    }
  }

  // Precondition: add_horizontal_layout > 1
  template <bool ConditionalAtomic, uint32_t LoopIndex>
  __device__ constexpr void add_patterns(uint32_t block_index,
                                         uint32_t lower_hash,
                                         uint32_t thread_index)
  {
    // Sanity check. TODO: remove redundant checks.
    static_assert(add_horizontal_layout > 1, "add_patterns() requires add_horizontal_layout > 1");

    if constexpr (LoopIndex < add_loop_count) {
      auto const pattern =
        policy_.template array_pattern<LoopIndex, add_horizontal_layout, add_vertical_layout>(
          lower_hash, thread_index);
      auto* word_base = words_ + block_index * words_per_block +
                        LoopIndex * add_vertical_layout * add_horizontal_layout +
                        thread_index * add_vertical_layout;

      for (int i = 0; i < add_vertical_layout; ++i) {
        atomic_or<ConditionalAtomic>(word_base + i, pattern[i]);
      }

      // Recurse.
      add_patterns<ConditionalAtomic, LoopIndex + 1>(block_index, lower_hash, thread_index);
    }
  }

  //===----------Cache-Sectorized Add----------===//
  template <bool ConditionalAtomic, uint32_t LoopIndex>
  __device__ constexpr void add_pattern_cs(uint32_t block_index,
                                           uint32_t lower_hash,
                                           uint32_t group_hash)
  {
    auto constexpr add_groups_per_vertical_layout = policy_type::add_groups_per_vertical_layout;
    auto constexpr group_index_width              = policy_type::group_index_width;
    auto constexpr group_index_mask               = policy_type::group_index_mask;
    auto constexpr words_per_group                = policy_type::words_per_group;

    // Sanity check. TODO: remove redundant checks.
    static_assert(add_horizontal_layout == 1, "add_pattern() requires add_horizontal_layout == 1");

    if constexpr (LoopIndex < add_loop_count) {
      auto const pattern =
        policy_.template array_pattern<LoopIndex, add_vertical_layout>(lower_hash);
      auto* word_base = words_ + block_index * words_per_block + LoopIndex * add_vertical_layout;

      for (int i = 0; i < add_groups_per_vertical_layout; ++i) {
        auto const group_index =
          (group_hash >> (i + LoopIndex * add_groups_per_vertical_layout) * group_index_width) &
          group_index_mask;
        atomic_or<ConditionalAtomic>(word_base + i * words_per_group + group_index, pattern[i]);
      }

      // Recurse.
      add_pattern_cs<ConditionalAtomic, LoopIndex + 1>(block_index, lower_hash, group_hash);
    }
  }

  template <bool ConditionalAtomic, uint32_t LoopIndex>
  __device__ constexpr void add_patterns_cs(uint32_t block_index,
                                            uint32_t lower_hash,
                                            uint32_t group_hash,
                                            uint32_t thread_index)
  {
    auto constexpr add_groups_per_vertical_layout = policy_type::add_groups_per_vertical_layout;
    auto constexpr group_index_width              = policy_type::group_index_width;
    auto constexpr group_index_mask               = policy_type::group_index_mask;
    auto constexpr words_per_group                = policy_type::words_per_group;

    // Sanity check. TODO: remove redundant checks.
    static_assert(add_horizontal_layout > 1, "add_patterns() requires add_horizontal_layout > 1");

    if constexpr (LoopIndex < add_loop_count) {
      auto const pattern =
        policy_.template array_pattern<LoopIndex, add_horizontal_layout, add_vertical_layout>(
          lower_hash, thread_index);
      auto* word_base = words_ + block_index * words_per_block +
                        LoopIndex * add_vertical_layout * add_horizontal_layout +
                        thread_index * add_vertical_layout;

      for (int i = 0; i < add_groups_per_vertical_layout; ++i) {
        auto const group_index =
          (group_hash >> (i + LoopIndex * add_groups_per_vertical_layout * add_horizontal_layout +
                          thread_index * add_groups_per_vertical_layout) *
                           group_index_width) &
          group_index_mask;
        atomic_or<ConditionalAtomic>(word_base + i * words_per_group + group_index, pattern[i]);
      }

      // Recurse.
      add_patterns_cs<ConditionalAtomic, LoopIndex + 1>(
        block_index, lower_hash, group_hash, thread_index);
    }
  }

  template <bool ConditionalAtomic>
  __device__ constexpr void atomic_or(word_type* word_ptr, word_type pattern) const
  {
    if constexpr (use_cuda_atomic_ref) {
      if constexpr (ConditionalAtomic) {
        if ((*word_ptr & pattern) != pattern) {
          auto atom_word = cuda::atomic_ref<word_type, thread_scope>{*word_ptr};
          atom_word.fetch_or(pattern, cuda::memory_order_relaxed);
        }
      } else {
        auto atom_word = cuda::atomic_ref<word_type, thread_scope>{*word_ptr};
        atom_word.fetch_or(pattern, cuda::memory_order_relaxed);
      }
    } else {
      if constexpr (ConditionalAtomic) {
        if ((*word_ptr & pattern) != pattern) {
          atomicOr(reinterpret_cast<atomic_word_type*>(word_ptr),
                   static_cast<atomic_word_type>(pattern));
        }
      } else {
        atomicOr(reinterpret_cast<atomic_word_type*>(word_ptr),
                 static_cast<atomic_word_type>(pattern));
      }
    }
  }

  /// Compare the stored pattern against the expected pattern for the given hash value.
  // Precondition: contains_horizontal_layout == 1
  template <uint32_t LoopIndex>
  __device__ constexpr bool compare_pattern(uint32_t block_index, uint32_t lower_hash) const
  {
    // Sanity check. TODO: remove redundant checks.
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
      if constexpr (use_early_exit) {
        if (!match) { return false; }
        return compare_pattern<LoopIndex + 1>(block_index, lower_hash);
      } else {
        return compare_pattern<LoopIndex + 1>(block_index, lower_hash) && match;
      }
    } else {
      return true;
    }
  }

  // Precondition: contains_horizontal_layout > 1
  template <uint32_t LoopIndex, class CG>
  __device__ constexpr bool compare_patterns(CG group,  // NOTE: this is only needed for early exit
                                             uint32_t block_index,
                                             uint32_t lower_hash,
                                             uint32_t thread_index) const
  {
    // Sanity check
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

      // Recurse.
      // Early exit in this implementation occurs at the granulairy of contains_vertical_layout
      // words.
      if constexpr (use_early_exit) {
        // This will degrade performance in selective contexts
        if (group.any(!match)) { return false; }
        return compare_patterns<LoopIndex + 1>(group, block_index, lower_hash, thread_index);
      } else {
        return compare_patterns<LoopIndex + 1>(group, block_index, lower_hash, thread_index) &&
               match;
      }
    } else {
      return true;
    }
  }

  //===----------Cache-Sectorized Compare----------===//
  template <uint32_t LoopIndex>
  __device__ constexpr bool compare_pattern_cs(uint32_t block_index,
                                               uint32_t lower_hash,
                                               uint32_t group_hash) const
  {
    auto constexpr contains_groups_per_vertical_layout =
      policy_type::contains_groups_per_vertical_layout;
    auto constexpr group_index_width = policy_type::group_index_width;
    auto constexpr group_index_mask  = policy_type::group_index_mask;
    auto constexpr words_per_group   = policy_type::words_per_group;

    // Sanity check. TODO: remove redundant checks.
    static_assert(contains_horizontal_layout == 1,
                  "compare_pattern() requires contains_horizontal_layout == 1");

    if constexpr (LoopIndex < contains_loop_count) {
      auto const* word_base =
        words_ + block_index * words_per_block + LoopIndex * contains_vertical_layout;
      auto const expected_pattern =
        policy_.template array_pattern<LoopIndex, contains_vertical_layout>(lower_hash);

      bool match = true;
      for (int i = 0; i < contains_groups_per_vertical_layout; ++i) {
        auto const group_index =
          (group_hash >>
           (i + LoopIndex * contains_groups_per_vertical_layout) * group_index_width) &
          group_index_mask;
        match &= (word_base[i * words_per_group + group_index] & expected_pattern[i]) ==
                 expected_pattern[i];
      }

      // Recurse.
      // Early exit in this implementation occurs at the granulairy of contains_vertical_layout
      // words.
      if constexpr (use_early_exit) {
        if (!match) { return false; }
        return compare_pattern_cs<LoopIndex + 1>(block_index, lower_hash, group_hash);
      } else {
        return compare_pattern_cs<LoopIndex + 1>(block_index, lower_hash, group_hash) && match;
      }
    } else {
      return true;
    }
  }

  template <uint32_t LoopIndex, class CG>
  __device__ constexpr bool compare_patterns_cs(
    CG group,  // NOTE: this is only needed for early exit
    uint32_t block_index,
    uint32_t lower_hash,
    uint32_t group_hash,
    uint32_t thread_index) const
  {
    auto constexpr contains_groups_per_vertical_layout =
      policy_type::contains_groups_per_vertical_layout;
    auto constexpr group_index_width = policy_type::group_index_width;
    auto constexpr group_index_mask  = policy_type::group_index_mask;
    auto constexpr words_per_group   = policy_type::words_per_group;

    // Sanity check
    static_assert(contains_horizontal_layout > 1,
                  "compare_patterns() requires HorizontalLayout > 1");

    if constexpr (LoopIndex < contains_loop_count) {
      auto const* word_base = words_ + block_index * words_per_block +
                              LoopIndex * contains_vertical_layout * contains_horizontal_layout +
                              thread_index * contains_vertical_layout;
      auto const expected_pattern =
        policy_
          .template array_pattern<LoopIndex, contains_horizontal_layout, contains_vertical_layout>(
            lower_hash, thread_index);

      bool match = true;
      for (int i = 0; i < contains_groups_per_vertical_layout; ++i) {
        auto const group_index =
          (group_hash >>
           (i + LoopIndex * contains_groups_per_vertical_layout * contains_horizontal_layout +
            thread_index * contains_groups_per_vertical_layout) *
             group_index_width) &
          group_index_mask;
        match &= (word_base[i * words_per_group + group_index] & expected_pattern[i]) ==
                 expected_pattern[i];
      }

      // Recurse.
      // Early exit in this implementation occurs at the granulairy of contains_vertical_layout
      // words.
      if constexpr (use_early_exit) {
        // This will degrade performance in selective contexts
        if (group.any(!match)) { return false; }
        return compare_patterns_cs<LoopIndex + 1>(
          group, block_index, lower_hash, group_hash, thread_index);
      } else {
        return compare_patterns_cs<LoopIndex + 1>(
                 group, block_index, lower_hash, group_hash, thread_index) &&
               match;
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
