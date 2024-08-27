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

#include <cuco/detail/bloom_filter/bloom_filter_impl.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/atomic>
#include <cuda/stream_ref>

#include <cstdint>

namespace cuco {

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
class bloom_filter_ref {
  using impl_type = detail::bloom_filter_impl<Key, Extent, Scope, Hash, BlockWords, Word>;

 public:
  static constexpr auto thread_scope = impl_type::thread_scope;
  static constexpr auto block_words  = impl_type::block_words;

  using key_type    = typename impl_type::key_type;  ///< Key Type
  using extent_type = typename impl_type::extent_type;
  using size_type   = typename extent_type::value_type;
  using hasher      = typename impl_type::hasher;
  using word_type   = typename impl_type::word_type;

  __host__ __device__ bloom_filter_ref(word_type* data,
                                       Extent num_blocks,
                                       std::uint32_t pattern_bits,
                                       cuda_thread_scope<Scope>,
                                       Hash const& hash);

  template <class CG>
  __device__ void clear(CG const& group);

  __host__ void clear(cuda::stream_ref stream = {});

  __host__ void clear_async(cuda::stream_ref stream = {});

  // TODO
  // template <class ProbeKey>
  // __device__ void add(ProbeKey const& key);

  template <class CG, class ProbeKey>
  __device__ void add(CG const& group, ProbeKey const& key);

  // TODO
  // template <class CG, class InputIt>
  // __device__ void add(CG const& group, InputIt first, InputIt last);

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

  template <class ProbeKey>
  [[nodiscard]] __device__ bool test(ProbeKey const& key) const;

  // TODO
  // template <class CG, class ProbeKey>
  // [[nodiscard]] __device__ bool test(CG const& group, ProbeKey const& key) const;

  // TODO
  // template <class CG, class InputIt, class OutputIt>
  // __device__ void test(CG const& group, InputIt first, InputIt last, OutputIt output_begin)
  // const;

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

  // TODO
  // __host__ __device__ word_type* data() const;
  // __host__ __device__ extent_type extent() const;
  // __host__ __device__ size_type num_blocks() const;
  // __host__ __device__ size_type num_words() const;
  // __host__ __device__ size_type num_bits() const;
  // __host__ __device__ hasher hash_function() const;
  // [[nodiscard]] __host__ float occupancy() const;
  // [[nodiscard]] __host__ float expected_false_positive_rate(size_t unique_keys) const
  // [[nodiscard]] __host__ __device__ static uint32_t optimal_pattern_bits(size_t num_blocks)

 private:
  impl_type impl_;
};
}  // namespace cuco

#include <cuco/detail/bloom_filter/bloom_filter_ref.inl>