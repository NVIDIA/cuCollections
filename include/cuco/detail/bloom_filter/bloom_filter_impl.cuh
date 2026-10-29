/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>
#include <cuco/detail/bloom_filter/kernels.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/utility/cuda.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/detail/utility/math.cuh>
#include <cuco/detail/utils.hpp>
#include <cuco/utility/cuda.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuco/utility/traits.hpp>

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/__memory/address_space.h>
#include <cuda/annotated_ptr>
#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>  // TODO #include <cuda/std/algorithm> once available
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <cuda/utility>

#include <cooperative_groups.h>

#include <cstddef>
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
  static_assert(
    cuda::std::is_same_v<word_type, unsigned int> ||
      cuda::std::is_same_v<word_type, unsigned long long int>,
    "Policy::word_type must be unsigned int or unsigned long long int for native atomicOr");

  static constexpr auto thread_scope    = Scope;
  static constexpr auto words_per_block = policy_type::words_per_block;

  static constexpr auto add_vertical_layout        = policy_type::add_vertical_layout;
  static constexpr auto add_horizontal_layout      = policy_type::add_horizontal_layout;
  static constexpr auto contains_vertical_layout   = policy_type::contains_vertical_layout;
  static constexpr auto contains_horizontal_layout = policy_type::contains_horizontal_layout;
  static constexpr bool conditional_add            = policy_type::conditional_add;
  static constexpr bool early_exit_contains        = policy_type::early_exit_contains;
  static constexpr bool persisting_l2_access       = policy_type::persisting_l2_access;
  static constexpr bool fast_l2_add                = policy_type::fast_l2_add;
  static constexpr auto add_keys_per_thread        = policy_type::add_keys_per_thread;
  static constexpr auto contains_keys_per_thread   = policy_type::contains_keys_per_thread;
  /// Per-thread salts of the add and contains layouts (one thread per key for horizontal layout 1)
  using add_salts_type =
    typename policy_type::template thread_salts<add_horizontal_layout, add_vertical_layout>;
  using contains_salts_type =
    typename policy_type::template thread_salts<contains_horizontal_layout,
                                                contains_vertical_layout>;
  static constexpr auto add_loop_count =
    words_per_block / (add_vertical_layout * add_horizontal_layout);
  static constexpr auto contains_loop_count =
    words_per_block / (contains_vertical_layout * contains_horizontal_layout);

  static_assert(cuda::std::has_single_bit(words_per_block) and words_per_block <= 32,
                "Number of words per block must be a power-of-two and less than or equal to 32");

  __host__ __device__ static constexpr size_t alignment() noexcept
  {
    // Maximum alignment is one sector.
    return cuda::std::min(
      static_cast<size_t>(cuco::utility::sector_size_bytes),
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
    auto const num_words = static_cast<cuco::detail::index_type>(num_blocks_) * words_per_block;
    auto const stride    = static_cast<cuco::detail::index_type>(group.size());
    for (cuco::detail::index_type i = static_cast<cuco::detail::index_type>(group.thread_rank());
         i < num_words;
         i += stride) {
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
      static_cast<cuco::detail::index_type>(num_blocks_) * words_per_block,
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
      static_cast<cuco::detail::index_type>(this->block_extent()) * words_per_block,
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
      static_cast<cuco::detail::index_type>(this->block_extent()) * words_per_block,
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

#pragma unroll
    for (uint32_t thread_index = 0; thread_index < add_horizontal_layout; ++thread_index) {
      add_patterns<ConditionalAdd, 0>(block_index, lower_hash, add_salts(thread_index));
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

      add_patterns<ConditionalAdd, 0>(block_index, lower_hash, add_salts(group.thread_rank()));
    }
  }

  template <bool ConditionalAdd = conditional_add, class CG, class BuildKey>
  __device__ void add_coop(CG group, BuildKey build_key)
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const [upper_hash, lower_hash] = policy_.split_hash(build_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);
    auto const thread_salts             = add_salts(group.thread_rank());

    if constexpr (fast_l2_add) {
      // Shuffle all keys' block indices and hashes first: the skipped word makes the atomics
      // per-lane conditional, and divergence ahead of a shuffle costs a convergence check per
      // shuffle. With the shuffles done up front a single check covers all of them.
      cuda::std::array<uint32_t, num_threads> blocks;
      cuda::std::array<uint32_t, num_threads> hashes;
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        blocks[i] = group.shfl(block_index, i);
        hashes[i] = group.shfl(lower_hash, i);
      }
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        add_patterns<ConditionalAdd, 0>(blocks[i], hashes[i], thread_salts);
      }
    } else {
#pragma unroll num_threads
      for (int i = 0; i < num_threads; ++i) {
        add_patterns<ConditionalAdd, 0>(
          group.shfl(block_index, i), group.shfl(lower_hash, i), thread_salts);
      }
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
    auto const thread_salts = add_salts(group.thread_rank());

    // Shuffle outside the per-key branch: the bulk kernels' tiles sync with the full-warp mask
#pragma unroll num_threads
    for (int i = 0; i < num_threads; ++i) {
      auto const valid = group.shfl(is_valid, i);
      auto const block = group.shfl(block_index, i);
      auto const hash  = group.shfl(lower_hash, i);
      if (valid) { add_patterns<ConditionalAdd, 0>(block, hash, thread_salts); }
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
    auto num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = 256;
    auto constexpr cg_size    = static_cast<int32_t>(add_horizontal_layout);

    if constexpr (add_keys_per_thread > 1) {
      // Full blocks of `add_keys_per_thread` keys per thread first; `add_n` takes the rest.
      auto constexpr keys_per_thread = static_cast<int32_t>(add_keys_per_thread);
      auto const kernel              = detail::bloom_filter_ns::add_n_multi_key<conditional_add,
                                                                                cg_size,
                                                                                block_size,
                                                                                keys_per_thread,
                                                                                InputIt,
                                                                                bloom_filter_impl>;
      auto const done =
        launch_multi_key<block_size, keys_per_thread>(kernel, num_keys, stream, first, *this);
      first += done;
      num_keys -= done;
      if (num_keys == 0) { return; }
    }
    detail::bloom_filter_ns::add_n<conditional_add, cg_size, block_size>
      <<<cuco::detail::int_div_ceil(num_keys, block_size), block_size, 0, stream.get()>>>(
        first, num_keys, *this);
  }

  template <class InputIt>
  __host__ void add(InputIt first, InputIt last, cuda::stream_ref stream)
  {
    this->add_async(first, last, stream);
    stream.sync();
  }

  template <class ProbeKey>
  __device__ bool contains(ProbeKey probe_key) const
  {
    auto const [upper_hash, lower_hash] = policy_.split_hash(probe_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);

    bool result = true;
#pragma unroll
    for (uint32_t thread_index = 0; thread_index < contains_horizontal_layout; ++thread_index) {
      result = result && compare_patterns<0>(block_index, lower_hash, contains_salts(thread_index));
    }
    return result;
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

      return group.all(
        compare_patterns<0>(block_index, lower_hash, contains_salts(group.thread_rank())));
    }
  }

  template <class CG, class ProbeKey>
  __device__ bool contains_coop(CG group, ProbeKey probe_key) const
  {
    constexpr auto num_threads = tile_size_v<CG>;

    auto const [upper_hash, lower_hash] = policy_.split_hash(probe_key);
    auto const block_index              = policy_.block_index(upper_hash, num_blocks_);
    auto const thread_salts             = contains_salts(group.thread_rank());
    bool result_out                     = false;

#pragma unroll num_threads
    for (int i = 0; i < num_threads; ++i) {
      auto const result = group.all(
        compare_patterns<0>(group.shfl(block_index, i), group.shfl(lower_hash, i), thread_salts));
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

    auto const thread_salts = contains_salts(group.thread_rank());
    bool result_out         = false;
    // An invalid key probes block 0 and its result is discarded, so that every thread reaches the
    // collectives (the bulk kernels' tiles sync with the full-warp mask)
#pragma unroll num_threads
    for (int i = 0; i < num_threads; ++i) {
      auto const result = group.all(
        compare_patterns<0>(group.shfl(block_index, i), group.shfl(lower_hash, i), thread_salts));
      if (i == group.thread_rank()) { result_out = result; }
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
    auto num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = 256;
    auto constexpr cg_size    = static_cast<int32_t>(contains_horizontal_layout);

    if constexpr (contains_keys_per_thread > 1) {
      // As in `add_async`: full multi-key blocks first, the remaining keys with `contains_n`.
      auto constexpr keys_per_thread = static_cast<int32_t>(contains_keys_per_thread);
      auto const kernel              = detail::bloom_filter_ns::contains_n_multi_key<cg_size,
                                                                                     block_size,
                                                                                     keys_per_thread,
                                                                                     InputIt,
                                                                                     OutputIt,
                                                                                     bloom_filter_impl>;
      auto const done                = launch_multi_key<block_size, keys_per_thread>(
        kernel, num_keys, stream, first, output_begin, *this);
      first += done;
      output_begin += done;
      num_keys -= done;
      if (num_keys == 0) { return; }
    }
    detail::bloom_filter_ns::contains_n<cg_size, block_size>
      <<<cuco::detail::int_div_ceil(num_keys, block_size), block_size, 0, stream.get()>>>(
        first, num_keys, output_begin, *this);
  }

  template <class InputIt, class OutputIt>
  __host__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         cuda::stream_ref stream) const
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
  __host__ void add_if(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream)
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
                            cuda::stream_ref stream) const
  {
    this->contains_if_async(first, last, stencil, pred, output_begin, stream);
    stream.sync();
  }

  //  private:
  template <class Pointer>
  [[nodiscard]] __device__ static Pointer filter_access(Pointer ptr) noexcept
  {
    if constexpr (persisting_l2_access) {
      return cuda::device::is_address_from(ptr, cuda::device::address_space::global)
               ? cuda::associate_access_property(ptr, cuda::access_property::persisting{})
               : ptr;
    } else {
      return ptr;
    }
  }

  // Launches the multi-key `kernel` on the full blocks of `BlockSize * KeysPerThread` keys and
  // returns the number of keys it covers: none for inputs below one wave of such blocks, which keep
  // one key per thread to use more SMs. The caller handles the remaining keys.
  template <int32_t BlockSize, int32_t KeysPerThread, class Kernel, class... Args>
  __host__ static cuco::detail::index_type launch_multi_key(Kernel kernel,
                                                            cuco::detail::index_type num_keys,
                                                            cuda::stream_ref stream,
                                                            Args... args)
  {
    auto constexpr keys_per_block = cuco::detail::index_type{BlockSize} * KeysPerThread;
    auto const num_blocks         = num_keys / keys_per_block;
    if (num_blocks == 0 or
        num_blocks < cuco::detail::max_occupancy_grid_size(BlockSize, kernel, stream)) {
      return 0;
    }
    kernel<<<num_blocks, BlockSize, 0, stream.get()>>>(args...);
    return num_blocks * keys_per_block;
  }

  template <uint32_t NumWords>
  __device__ constexpr cuda::std::array<word_type, NumWords> vec_load_words(
    cuco::detail::index_type index) const
  {
    // The block storage is aligned to `alignment()`, but a per-lane load at offset `index` is
    // only guaranteed to be aligned to `min(NumWords * sizeof(word_type), alignment())`. Hand the
    // compiler the alignment that's actually delivered, not the block-level maximum.
    constexpr auto load_alignment =
      cuda::std::min<size_t>(NumWords * sizeof(word_type), alignment());
    auto const* ptr =
      reinterpret_cast<cuda::std::array<word_type, NumWords> const*>(words_ + index);
    return *static_cast<cuda::std::array<word_type, NumWords> const*>(
      __builtin_assume_aligned(filter_access(ptr), load_alignment));
  }

  // Salts of a thread of the group cooperating on a key, selected once per thread and reused for
  // every key. With horizontal layout 1 the only rank is 0 and the salts are compile-time
  // constants.
  __device__ constexpr add_salts_type add_salts(uint32_t thread_index) const
  {
    return policy_.template make_thread_salts<add_horizontal_layout, add_vertical_layout>(
      thread_index);
  }

  __device__ constexpr contains_salts_type contains_salts(uint32_t thread_index) const
  {
    return policy_.template make_thread_salts<contains_horizontal_layout, contains_vertical_layout>(
      thread_index);
  }

  template <bool ConditionalAdd, uint32_t LoopIndex>
  __device__ constexpr void add_patterns(uint32_t block_index,
                                         uint32_t lower_hash,
                                         add_salts_type const& thread_salts)
  {
    if constexpr (LoopIndex < add_loop_count) {
      // A FastL2Add thread that owns the whole first sector (horizontal layout 1) issues one atomic
      // less, shifting past the skipped word, instead of a conditional atomic per word.
      constexpr uint32_t sector_words   = 32 / sizeof(word_type);
      constexpr bool shift_past_skipped = fast_l2_add and add_horizontal_layout == 1 and
                                          LoopIndex == 0 and add_vertical_layout >= sector_words;
      auto const pattern =
        policy_.template array_pattern<LoopIndex, not shift_past_skipped>(lower_hash, thread_salts);
      auto* word_base = words_ +
                        static_cast<cuco::detail::index_type>(block_index) * words_per_block +
                        LoopIndex * add_vertical_layout * add_horizontal_layout +
                        thread_salts.thread_index * add_vertical_layout;

      if constexpr (shift_past_skipped) {
        uint32_t const skip = policy_type::skip_index(lower_hash);
        for (uint32_t i = 0; i + 1 < sector_words; ++i) {
          bool const past = i >= skip;
          atomic_or<ConditionalAdd, false>(word_base + i + past,
                                           past ? pattern[i + 1] : pattern[i]);
        }
        for (uint32_t i = sector_words; i < add_vertical_layout; ++i) {
          atomic_or<ConditionalAdd, false>(word_base + i, pattern[i]);
        }
      } else {
        for (int i = 0; i < add_vertical_layout; ++i) {
          atomic_or<ConditionalAdd>(word_base + i, pattern[i]);
        }
      }

      // Recurse.
      add_patterns<ConditionalAdd, LoopIndex + 1>(block_index, lower_hash, thread_salts);
    }
  }

  template <bool ConditionalAdd, bool SkipZero = fast_l2_add>
  __device__ constexpr void atomic_or(word_type* word_ptr, word_type pattern) const
  {
    // Issue no atomic for a zero pattern: with FastL2Add the skipped word's pattern is cleared, so
    // the coalesced request leaves that word inactive.
    if constexpr (SkipZero) {
      if (pattern == 0) { return; }
    }
    auto const do_or = [&]() {
      auto* const p = filter_access(word_ptr);
      if constexpr (thread_scope == cuda::thread_scope_thread) {
        *p |= pattern;
      } else if constexpr (thread_scope == cuda::thread_scope_block) {
        atomicOr_block(p, pattern);
      } else if constexpr (thread_scope == cuda::thread_scope_device) {
        atomicOr(p, pattern);
      } else if constexpr (thread_scope == cuda::thread_scope_system) {
        atomicOr_system(p, pattern);
      } else {
        static_assert(cuco::dependent_false<word_type>,
                      "unsupported cuda::thread_scope for native atomic_or");
      }
    };

    if constexpr (ConditionalAdd) {
      // Benign non-atomic read racing with atomicOr; technically UB but used throughout cuco.
      if ((*filter_access(word_ptr) & pattern) != pattern) { do_or(); }
    } else {
      do_or();
    }
  }

  /// Whether every stored word contains its expected bits. The FastL2Add skipped word (unset by
  /// `add`) passes by its index, so every word keeps a plain bit test.
  template <class PatternArrayT>
  __device__ constexpr bool matches(PatternArrayT const& stored_pattern,
                                    PatternArrayT const& expected_pattern,
                                    uint32_t lower_hash,
                                    uint32_t first_word) const
  {
    bool match = true;
    for (uint32_t i = 0; i < expected_pattern.size(); ++i) {
      match &= (stored_pattern[i] & expected_pattern[i]) == expected_pattern[i] or
               (fast_l2_add and first_word + i == policy_type::skip_index(lower_hash));
    }
    return match;
  }

  /// Compare the stored pattern against the expected pattern for the given hash value.
  template <uint32_t LoopIndex>
  __device__ constexpr bool compare_patterns(uint32_t block_index,
                                             uint32_t lower_hash,
                                             contains_salts_type const& thread_salts) const
  {
    if constexpr (LoopIndex < contains_loop_count) {
      auto const stored_pattern = this->vec_load_words<contains_vertical_layout>(
        static_cast<cuco::detail::index_type>(block_index) * words_per_block +
        LoopIndex * contains_vertical_layout * contains_horizontal_layout +
        thread_salts.thread_index * contains_vertical_layout);
      auto const expected_pattern =
        policy_.template array_pattern<LoopIndex, false>(lower_hash, thread_salts);
      bool const match =
        matches(stored_pattern,
                expected_pattern,
                lower_hash,
                (LoopIndex * contains_horizontal_layout + thread_salts.thread_index) *
                  contains_vertical_layout);

      // Per-thread early exit: short-circuit this thread's recursion if its slice already missed.
      if constexpr (early_exit_contains) {
        if (!match) { return false; }
        return compare_patterns<LoopIndex + 1>(block_index, lower_hash, thread_salts);
      } else {
        return compare_patterns<LoopIndex + 1>(block_index, lower_hash, thread_salts) && match;
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
