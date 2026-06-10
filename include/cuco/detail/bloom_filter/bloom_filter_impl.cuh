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

  static constexpr auto thread_scope    = Scope;
  static constexpr auto words_per_block = policy_type::words_per_block;

  __host__ __device__ static constexpr size_t max_vec_bytes() noexcept
  {
    constexpr auto word_bytes  = sizeof(word_type);
    constexpr auto block_bytes = word_bytes * words_per_block;
    return cuda::std::min(cuda::std::max(word_bytes, 32ul),
                          block_bytes);  // aiming for 2xLDG128 -> 1 sector per thread
  }

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
  __device__ constexpr void clear(CG group)
  {
    for (int i = group.thread_rank(); i < num_blocks_ * words_per_block; i += group.size()) {
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

  template <class HashValue, class BlockIndex>
  __device__ void add_impl(HashValue const& hash_value, BlockIndex block_index)
  {
    add_impl_functor<HashValue, BlockIndex, policy_type, word_type, thread_scope> functor{
      hash_value, block_index, policy_, words_, words_per_block};
    cuda::static_for<words_per_block>(functor);
  }

  template <class CG, class ProbeKey>
  __device__ void add(CG group, ProbeKey const& key)
  {
    constexpr auto num_threads         = tile_size_v<CG>;
    constexpr auto optimal_num_threads = add_optimal_cg_size();
    constexpr auto worker_num_threads =
      (num_threads < optimal_num_threads) ? num_threads : optimal_num_threads;

    // If single thread is optimal, use scalar add
    if constexpr (worker_num_threads == 1) {
      this->add(key);
    } else {
      auto const hash_value = policy_.hash(key);
      this->add_impl(hash_value, policy_.block_index(hash_value, num_blocks_));
    }
  }

  template <class CG, class InputIt>
  __device__ void add(CG group, InputIt first, InputIt last)
  {
    namespace cg = cooperative_groups;

    constexpr auto num_threads         = tile_size_v<CG>;
    constexpr auto optimal_num_threads = add_optimal_cg_size();
    constexpr auto worker_num_threads =
      (num_threads < optimal_num_threads) ? num_threads : optimal_num_threads;

    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto const rank = group.thread_rank();

    // If single thread is optimal, use scalar add
    if constexpr (worker_num_threads == 1) {
      for (auto i = rank; i < num_keys; i += num_threads) {
        typename cuda::std::iterator_traits<InputIt>::value_type const& insert_element{
          *(first + i)};
        this->add(insert_element);
      }
    } else if constexpr (num_threads == worker_num_threads) {  // given CG is optimal CG
      typename policy_type::hash_result_type hash_value;
      size_type block_index;

      auto const group_iters = cuco::detail::int_div_ceil(num_keys, num_threads);
      for (size_type i = 0; (i / num_threads) < group_iters; i += num_threads) {
        if (i + rank < num_keys) {
          typename cuda::std::iterator_traits<InputIt>::value_type const& insert_element{
            *(first + i + rank)};
          hash_value  = policy_.hash(insert_element);
          block_index = policy_.block_index(hash_value, num_blocks_);
        }

        add_group_functor<CG, decltype(hash_value), decltype(block_index), bloom_filter_impl>
          functor{group,
                  hash_value,
                  block_index,
                  static_cast<size_t>(i),
                  static_cast<size_t>(num_keys),
                  static_cast<size_t>(num_threads),
                  this};
        cuda::static_for<num_threads>(functor);
      }
    } else {  // subdivide given CG into multiple optimal CGs
      typename policy_type::hash_result_type hash_value;
      size_type block_index;

      auto const worker_group  = cg::tiled_partition<worker_num_threads, CG>(group);
      auto const worker_offset = worker_num_threads * worker_group.meta_group_rank();

      auto const group_iters = cuco::detail::int_div_ceil(num_keys, num_threads);

      for (size_type i = 0; (i / num_threads) < group_iters; i += num_threads) {
        if (i + rank < num_keys) {
          typename cuda::std::iterator_traits<InputIt>::value_type const& key{*(first + i + rank)};
          hash_value  = policy_.hash(key);
          block_index = policy_.block_index(hash_value, num_blocks_);
        }

        add_worker_group_functor<decltype(worker_group),
                                 decltype(hash_value),
                                 decltype(block_index),
                                 bloom_filter_impl>
          functor{worker_group,
                  hash_value,
                  block_index,
                  static_cast<size_t>(i),
                  static_cast<size_t>(worker_offset),
                  static_cast<size_t>(num_keys),
                  static_cast<size_t>(worker_num_threads),
                  this};
        cuda::static_for<worker_num_threads>(functor);
      }
    }
  }

  template <class CG, class HashValue, class BlockIndex>
  __device__ void add_impl(CG group, HashValue const& hash_value, BlockIndex block_index)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const rank = group.thread_rank();

    if constexpr (num_threads == words_per_block) {
      auto atom_word = cuda::atomic_ref<word_type, thread_scope>{
        *(words_ + (block_index * words_per_block + rank))};
      atom_word.fetch_or(policy_.word_pattern(hash_value, rank), cuda::memory_order_relaxed);
    } else {
      add_impl_group_functor<HashValue, BlockIndex, word_type, thread_scope, policy_type> functor{
        hash_value, block_index, words_, words_per_block, rank, num_threads, policy_};
      cuda::static_for<words_per_block>(functor);
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

    bool result = true;
    contains_functor<decltype(hash_value), decltype(stored_pattern), policy_type> functor{
      hash_value, stored_pattern, policy_, &result};
    cuda::static_for<words_per_block>(functor);
    if (!result) { return false; }

    return true;
  }

  template <class CG, class ProbeKey>
  [[nodiscard]] __device__ bool contains(CG group, ProbeKey const& key) const
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

// Use pragma unroll instead of cuda::static_for to avoid CUDA 12.0 compatibility issues
#pragma unroll
      for (size_type i = 0; i < optimal_num_threads; ++i) {
        if (i >= rank && (i - rank) % num_threads == 0) {
          auto const thread_offset  = i * words_per_thread;
          auto const stored_pattern = this->vec_load_words<words_per_thread>(
            policy_.block_index(hash_value, num_blocks_) * words_per_block + thread_offset);

#pragma unroll
          for (size_type j = 0; j < words_per_thread; ++j) {
            auto const expected_pattern = policy_.word_pattern(hash_value, thread_offset + j);
            if ((stored_pattern[j] & expected_pattern) != expected_pattern) { success = false; }
          }
        }
      }

      return group.all(success);
    }
  }

  // TODO
  // template <class CG, class InputIt, class OutputIt>
  // __device__ void contains(CG group, InputIt first, InputIt last, OutputIt output_begin)
  // const;

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
    auto const always_true = cuda::constant_iterator<bool>{true};
    this->contains_if_async(first, last, always_true, cuda::std::identity{}, output_begin, stream);
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

    auto constexpr cg_size    = contains_optimal_cg_size();
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

  // TODO
  // [[nodiscard]] __host__ double occupancy() const;
  // [[nodiscard]] __host__ double expected_false_positive_rate(size_t unique_keys) const
  // [[nodiscard]] __host__ __device__ static uint32_t optimal_pattern_bits(size_t num_blocks)
  // template <typename CG, cuda::thread_scope NewScope = thread_scope>
  // [[nodiscard]] __device__ constexpr auto make_copy(CG group, word_type* const
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
