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

  // Implementation-tuning knobs. Not part of the public API; reached via
  // `bloom_filter_impl::tuning::use_*` from internal kernels. Defaults reflect the ablation
  // measurements from arXiv:2512.15595; flip in source for tuning experiments.
  struct tuning {
    static constexpr bool use_invoke_one                       = true;
    static constexpr bool use_early_exit                       = false;
    static constexpr bool use_cub_kernels                      = true;
    static constexpr bool use_warp_cooperative_add_kernel      = true;
    static constexpr bool use_warp_cooperative_contains_kernel = true;
    static constexpr bool use_work_stealing_add_kernel         = false;
    static constexpr bool use_work_stealing_contains_kernel    = false;
    static constexpr bool use_cuda_atomic_ref                  = false;
  };

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

  static constexpr bool is_cache_sectorized = policy_type::is_cache_sectorized;

  static_assert((not tuning::use_cuda_atomic_ref) or
                  (Scope == cuda::thread_scope::thread_scope_device),
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

  // Single Thread Add
  template <bool ConditionalAtomic = true, class BuildKey>
  __device__ void add(BuildKey build_key)
  {
    static_assert(add_horizontal_layout == 1, "This add() requires add_horizontal_layout == 1");

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
  template <bool ConditionalAtomic = true, class CG, class BuildKey>
  __device__ void add(CG group, BuildKey build_key)
  {
    static_assert(add_horizontal_layout > 1, "This add() requires add_horizontal_layout > 1");

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
  template <bool ConditionalAtomic = true, class CG, class BuildKey>
  __device__ void add_coop(CG group, BuildKey build_key)
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

  template <bool ConditionalAtomic = true, class CG, class BuildKey>
  __device__ void add_coop(CG group, BuildKey build_key, bool is_valid)
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

  // Device-side range add (single-thread): loops over keys.
  template <class InputIt>
  __device__ void add(InputIt first, InputIt last)
  {
    auto const num_keys = cuco::detail::distance(first, last);
    for (decltype(num_keys) i = 0; i < num_keys; ++i) {
      this->add(*(first + i));
    }
  }

  // Device-side range add (cooperative): each iteration cooperatively inserts one key.
  template <class CG, class InputIt>
  __device__ void add(CG group, InputIt first, InputIt last)
  {
    auto const num_keys = cuco::detail::distance(first, last);
    for (decltype(num_keys) i = 0; i < num_keys; ++i) {
      this->add(group, *(first + i));
    }
  }

  // Host-side Add Entry Points
  template <class InputIt>
  __host__ void add_async(InputIt first, InputIt last, cuda::stream_ref stream) noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = 256;
    auto constexpr cg_size    = static_cast<int32_t>(add_horizontal_layout);
    auto const grid_size      = tuning::use_warp_cooperative_add_kernel
                                  ? cuco::detail::int_div_ceil(num_keys, block_size)
                                  : cuco::detail::int_div_ceil(num_keys * cg_size, block_size);
    auto const l2_cache_size  = static_cast<size_t>(cuco::detail::l2_cache_size());
    auto const filter_size    = static_cast<size_t>(static_cast<size_type>(num_blocks_)) *
                             words_per_block * sizeof(word_type);

    if (2 * filter_size < l2_cache_size) {
      if constexpr (tuning::use_work_stealing_add_kernel) {
        detail::bloom_filter_ns::add_work_stealing_n<false, cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, *this);
      } else {
        detail::bloom_filter_ns::add_n<false, cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, *this);
      }
    } else {
      if constexpr (tuning::use_work_stealing_add_kernel) {
        detail::bloom_filter_ns::add_work_stealing_n<true, cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, *this);
      } else {
        detail::bloom_filter_ns::add_n<true, cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, *this);
      }
    }
  }

  template <class InputIt>
  __host__ void add(InputIt first, InputIt last, cuda::stream_ref stream) noexcept
  {
    this->add_async(first, last, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  // Single Thread Contains
  template <class ProbeKey>
  __device__ bool contains(ProbeKey probe_key) const
  {
    static_assert(contains_horizontal_layout == 1,
                  "This contains() requires contains_horizontal_layout == 1");

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
  __device__ bool contains(CG group, ProbeKey probe_key) const
  {
    static_assert(contains_horizontal_layout > 1,
                  "This contains() requires contains_horizontal_layout > 1");
    static_assert(tile_size_v<CG> == contains_horizontal_layout,
                  "This contains() requires CG with size equal to contains_horizontal_layout");

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
  __device__ bool contains_coop(CG group, ProbeKey probe_key) const
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
  __device__ bool contains_coop(CG group, ProbeKey probe_key, bool is_valid) const
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

  // Device-side range contains (single-thread): loops over keys.
  template <class InputIt, class OutputIt>
  __device__ void contains(InputIt first, InputIt last, OutputIt output_begin) const
  {
    auto const num_keys = cuco::detail::distance(first, last);
    for (decltype(num_keys) i = 0; i < num_keys; ++i) {
      *(output_begin + i) = this->contains(*(first + i));
    }
  }

  // Device-side range contains (cooperative): each iteration cooperatively queries one key.
  template <class CG, class InputIt, class OutputIt>
  __device__ void contains(CG group, InputIt first, InputIt last, OutputIt output_begin) const
  {
    auto const num_keys = cuco::detail::distance(first, last);
    for (decltype(num_keys) i = 0; i < num_keys; ++i) {
      auto const found = this->contains(group, *(first + i));
      if (group.thread_rank() == 0) { *(output_begin + i) = found; }
    }
  }

  // Host-side Contains Entry Points
  template <class InputIt, class OutputIt>
  __host__ void contains_async(InputIt first,
                               InputIt last,
                               OutputIt output_begin,
                               cuda::stream_ref stream) const noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    if constexpr (tuning::use_cub_kernels and ((words_per_block / contains_vertical_layout) == 1)) {
      cub::DeviceTransform::Transform(
        first,
        output_begin,
        num_keys,
        [*this] __device__(auto const& key) { return this->contains(key); },
        stream.get());
    } else {
      auto constexpr block_size = 256;
      auto constexpr cg_size    = static_cast<int32_t>(contains_horizontal_layout);
      auto const grid_size      = tuning::use_warp_cooperative_contains_kernel
                                    ? cuco::detail::int_div_ceil(num_keys, block_size)
                                    : cuco::detail::int_div_ceil(num_keys * cg_size, block_size);

      if constexpr (tuning::use_work_stealing_contains_kernel) {
        detail::bloom_filter_ns::contains_work_stealing_n<cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, output_begin, *this);
      } else {
        detail::bloom_filter_ns::contains_n<cg_size, block_size>
          <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, output_begin, *this);
      }
    }
  }

  template <class InputIt, class OutputIt>
  __host__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         cuda::stream_ref stream) const noexcept
  {
    this->contains_async(first, last, output_begin, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  // Host-side stencil-gated Add Entry Points
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
    auto const grid_size      = tuning::use_warp_cooperative_add_kernel
                                  ? cuco::detail::int_div_ceil(num_keys, block_size)
                                  : cuco::detail::int_div_ceil(num_keys * cg_size, block_size);
    auto const l2_cache_size  = static_cast<size_t>(cuco::detail::l2_cache_size());
    auto const filter_size    = static_cast<size_t>(static_cast<size_type>(num_blocks_)) *
                             words_per_block * sizeof(word_type);

    if (2 * filter_size < l2_cache_size) {
      detail::bloom_filter_ns::add_if_n<false, cg_size, block_size>
        <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, stencil, pred, *this);
    } else {
      detail::bloom_filter_ns::add_if_n<true, cg_size, block_size>
        <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, stencil, pred, *this);
    }
  }

  template <class InputIt, class StencilIt, class Predicate>
  __host__ void add_if(InputIt first,
                       InputIt last,
                       StencilIt stencil,
                       Predicate pred,
                       cuda::stream_ref stream) noexcept
  {
    this->add_if_async(first, last, stencil, pred, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  // Host-side stencil-gated Contains Entry Points
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
    auto const grid_size      = tuning::use_warp_cooperative_contains_kernel
                                  ? cuco::detail::int_div_ceil(num_keys, block_size)
                                  : cuco::detail::int_div_ceil(num_keys * cg_size, block_size);

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
    if constexpr (tuning::use_cuda_atomic_ref) {
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
      if constexpr (tuning::use_early_exit) {
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
      if constexpr (tuning::use_early_exit) {
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
      if constexpr (tuning::use_early_exit) {
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
      if constexpr (tuning::use_early_exit) {
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
