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
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/allocator.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/atomic>
#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cuco {

template <class Key,
          class Extent             = cuco::extent<std::size_t>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Hash               = cuco::xxhash_64<Key>,
          class Allocator          = cuco::cuda_allocator<std::byte>,
          std::uint32_t BlockWords = 8,
          class Word               = std::uint32_t>
class bloom_filter {
 public:
  template <cuda::thread_scope NewScope = Scope>
  using ref_type = bloom_filter_ref<Key, Extent, NewScope, Hash, BlockWords, Word>;

  static constexpr auto thread_scope = ref_type<>::thread_scope;
  static constexpr auto block_words  = ref_type<>::block_words;

  using key_type    = typename ref_type<>::key_type;  ///< Key Type
  using extent_type = typename ref_type<>::extent_type;
  using size_type   = typename extent_type::value_type;
  using hasher      = typename ref_type<>::hasher;
  using word_type   = typename ref_type<>::word_type;
  using allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<word_type>;

  __host__ bloom_filter(Extent num_blocks,
                        std::uint32_t pattern_bits,
                        cuda_thread_scope<Scope> = {},
                        Hash const& hash         = {},
                        Allocator const& alloc   = {},
                        cuda::stream_ref stream  = {});

  __host__ void clear(cuda::stream_ref stream = {});

  __host__ void clear_async(cuda::stream_ref stream = {});

  template <class InputIt>
  __host__ void add(InputIt first, InputIt last, cuda::stream_ref stream = {});

  template <class InputIt>
  __host__ void add_async(InputIt first, InputIt last, cuda::stream_ref stream = {});

  template <class InputIt, class StencilIt, class Predicate>
  __host__ void add_if(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream = {});

  template <class InputIt, class StencilIt, class Predicate>
  __host__ void add_if_async(InputIt first,
                             InputIt last,
                             StencilIt stencil,
                             Predicate pred,
                             cuda::stream_ref stream = {}) noexcept;

  template <class InputIt, class OutputIt>
  __host__ void test(InputIt first,
                     InputIt last,
                     OutputIt output_begin,
                     cuda::stream_ref stream = {}) const;

  template <class InputIt, class OutputIt>
  __host__ void test_async(InputIt first,
                           InputIt last,
                           OutputIt output_begin,
                           cuda::stream_ref stream = {}) const noexcept;

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void test_if(InputIt first,
                        InputIt last,
                        StencilIt stencil,
                        Predicate pred,
                        OutputIt output_begin,
                        cuda::stream_ref stream = {}) const;

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void test_if_async(InputIt first,
                              InputIt last,
                              StencilIt stencil,
                              Predicate pred,
                              OutputIt output_begin,
                              cuda::stream_ref stream = {}) const noexcept;

  [[nodiscard]] __host__ word_type* data() noexcept;

  [[nodiscard]] __host__ word_type const* data() const noexcept;

  [[nodiscard]] __host__ extent_type block_extent() const noexcept;

  [[nodiscard]] __host__ hasher hash_function() const noexcept;

  [[nodiscard]] __host__ allocator_type allocator() const noexcept;

  [[nodiscard]] ref_type<> ref() const noexcept;

 private:
  allocator_type allocator_;
  std::unique_ptr<word_type, detail::custom_deleter<std::size_t, allocator_type>> data_;
  ref_type<> ref_;
};

}  // namespace cuco

#include <cuco/detail/bloom_filter/bloom_filter.inl>