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

#include <cuco/bloom_filter_ref.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/storage.cuh>
#include <cuco/types.cuh>
#include <cuco/utility/allocator.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuco/utility/traits.hpp>

#include <cuda/std/atomic>
#include <cuda/stream_ref>

#include <cstddef>
#include <memory>

// move these includes to .inl
#include <cuco/detail/bloom_filter/kernels.cuh>

#include <cub/device/device_for.cuh>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

namespace cuco {

template <class Key,
          class Extent             = cuco::extent<size_t>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Hash               = cuco::default_hash_function<Key>,
          class Allocator          = cuco::cuda_allocator<std::byte>,
          class Storage            = cuco::storage<1>>
class bloom_filter {
 public:
  static constexpr auto window_size  = Storage::window_size;  ///< Window size used for probing
  static constexpr auto thread_scope = Scope;                 ///< CUDA thread scope

  using key_type    = Key;
  using word_type   = uint32_t;
  using extent_type = Extent;
  using size_type   = typename extent_type::value_type;  ///< Size type
  using storage_type =
    detail::storage<Storage, word_type, extent_type, Allocator>;   ///< Storage type
  using allocator_type   = typename storage_type::allocator_type;  ///< Allocator type
  using storage_ref_type = typename storage_type::ref_type;  ///< Non-owning window storage ref type

  template <typename... Operators>
  using ref_type = cuco::bloom_filter_ref<key_type,
                                          thread_scope,
                                          Hash,
                                          storage_ref_type,
                                          Operators...>;  ///< Non-owning container ref type

  bloom_filter(bloom_filter const&)            = delete;
  bloom_filter& operator=(bloom_filter const&) = delete;

  bloom_filter(bloom_filter&&) = default;  ///< Move constructor

  /**
   * @brief Replaces the contents of the container with another container.
   *
   * @return Reference of the current set object
   */
  bloom_filter& operator=(bloom_filter&&) = default;
  ~bloom_filter()                         = default;

  constexpr bloom_filter(Extent num_sub_filters,
                         uint32_t pattern_bits,
                         cuda_thread_scope<Scope> = {},
                         Hash hash                = {},
                         Storage                  = {},
                         Allocator const& alloc   = {},
                         cuda::stream_ref stream  = {})
    : pattern_bits_{pattern_bits},
      hash_{hash},
      storage_{std::make_unique<storage_type>(num_sub_filters, alloc)}
  {
    this->clear_async(stream);
  }

  void clear(cuda::stream_ref stream = {}) { this->storage_->initialize(word_type{}, stream); }

  void clear_async(cuda::stream_ref stream = {}) noexcept
  {
    this->storage_->initialize_async(word_type{}, stream);
  }

  template <class InputIt>
  void add(InputIt first, InputIt last, cuda::stream_ref stream = {})
  {
    this->add_async(first, last, stream);
    stream.wait();
  }

  template <class InputIt>
  void add_async(InputIt first, InputIt last, cuda::stream_ref stream = {})
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    if constexpr (window_size == 1) {
      auto add_op = [ref = this->ref(op::add)] __device__(key_type const key) mutable {
        ref.add(key);
      };
      CUCO_CUDA_TRY(cub::DeviceFor::ForEachCopyN(first, num_keys, add_op, stream.get()));
    } else {
      auto const always_true = thrust::constant_iterator<bool>{true};
      this->add_if_async(first, last, always_true, thrust::identity{}, stream);
    }
  }

  template <class InputIt, class StencilIt, class Predicate>
  void add_if(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream = {})
  {
    this->add_if_async(first, last, stencil, pred, stream);
    stream.wait();
  }

  template <class InputIt, class StencilIt, class Predicate>
  void add_if_async(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream = {})
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = cuco::detail::default_block_size();
    auto const grid_size =
      cuco::detail::grid_size(num_keys, window_size, cuco::detail::default_stride(), block_size);

    bloom_filter_ns::detail::add_if_n<block_size><<<grid_size, block_size, 0, stream.get()>>>(
      first, num_keys, stencil, pred, this->ref(op::add));
  }

  template <class InputIt, class OutputIt>
  void contains(InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream = {})
  {
    this->contains_async(first, last, output_begin, stream);
    stream.wait();
  }

  template <class InputIt, class OutputIt>
  void contains_async(InputIt first,
                      InputIt last,
                      OutputIt output_begin,
                      cuda::stream_ref stream = {})
  {
    auto const always_true = thrust::constant_iterator<bool>{true};
    this->contains_if_async(first, last, always_true, thrust::identity{}, output_begin, stream);
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  void contains_if(InputIt first,
                   InputIt last,
                   StencilIt stencil,
                   Predicate pred,
                   OutputIt output_begin,
                   cuda::stream_ref stream = {})
  {
    this->contains_if_async(first, last, stencil, pred, output_begin, stream);
    stream.wait();
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  void contains_if_async(InputIt first,
                         InputIt last,
                         StencilIt stencil,
                         Predicate pred,
                         OutputIt output_begin,
                         cuda::stream_ref stream = {})
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = cuco::detail::default_block_size();
    auto const grid_size =
      cuco::detail::grid_size(num_keys, 1, cuco::detail::default_stride(), block_size);

    bloom_filter_ns::detail::contains_if_n<block_size><<<grid_size, block_size, 0, stream.get()>>>(
      first, num_keys, stencil, pred, output_begin, this->ref(op::contains));
  }

  template <typename... Operators>
  [[nodiscard]] auto ref(Operators...) const noexcept
  {
    static_assert(sizeof...(Operators), "No operators specified");
    return ref_type<Operators...>(
      pattern_bits_, cuda_thread_scope<Scope>{}, hash_, storage_->ref());
  }

 private:
  uint32_t pattern_bits_;
  Hash hash_;
  std::unique_ptr<storage_type> storage_;
};

}  // namespace cuco