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

#include <cuco/cuda_stream_ref.hpp>
#include <cuco/detail/hyperloglog/hyperloglog.cuh>
#include <cuco/distinct_count_estimator_ref.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/allocator.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cstddef>
#include <iterator>
#include <memory>

namespace cuco {
template <class T,
          int32_t Precision        = 11,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Hash               = cuco::xxhash_64<T>,
          class Allocator          = cuco::cuda_allocator<std::byte>>
class distinct_count_estimator {
  using impl_type = detail::hyperloglog<T, Precision, Scope, Hash, Allocator>;

 public:
  static constexpr auto thread_scope = impl_type::thread_scope;  ///< CUDA thread scope
  static constexpr auto precision    = impl_type::precision;

  using allocator_type = typename impl_type::allocator_type;  ///< Allocator type
  using storage_type   = typename impl_type::storage_type;

  template <cuda::thread_scope NewScope = thread_scope>
  using ref_type = cuco::distinct_count_estimator_ref<T, Precision, NewScope, Hash>;

  // TODO enable CTAD
  constexpr distinct_count_estimator(cuco::cuda_thread_scope<Scope> scope = {},
                                     Hash const& hash                     = {},
                                     Allocator const& alloc               = {},
                                     cuco::cuda_stream_ref stream         = {});

  distinct_count_estimator(distinct_count_estimator const&) = delete;
  distinct_count_estimator& operator=(distinct_count_estimator const&) = delete;
  distinct_count_estimator(distinct_count_estimator&&)                 = default;
  distinct_count_estimator& operator=(distinct_count_estimator&&) = default;
  ~distinct_count_estimator()                                     = default;

  void clear_async(cuco::cuda_stream_ref stream = {}) noexcept;

  void clear(cuco::cuda_stream_ref stream = {});

  template <class InputIt>
  void add_async(InputIt first, InputIt last, cuco::cuda_stream_ref stream = {}) noexcept;

  template <class InputIt>
  void add(InputIt first, InputIt last, cuco::cuda_stream_ref stream = {});

  template <cuda::thread_scope OtherScope, class OtherAllocator>
  void merge_async(
    distinct_count_estimator<T, Precision, OtherScope, Hash, OtherAllocator> const& other,
    cuco::cuda_stream_ref stream = {}) noexcept;

  template <cuda::thread_scope OtherScope, class OtherAllocator>
  void merge(distinct_count_estimator<T, Precision, OtherScope, Hash, OtherAllocator> const& other,
             cuco::cuda_stream_ref stream = {});

  template <cuda::thread_scope OtherScope>
  void merge_async(ref_type<OtherScope> const& other, cuco::cuda_stream_ref stream = {}) noexcept;

  template <cuda::thread_scope OtherScope>
  void merge(ref_type<OtherScope> const& other, cuco::cuda_stream_ref stream = {});

  [[nodiscard]] std::size_t estimate(cuco::cuda_stream_ref stream = {}) const;

  [[nodiscard]] ref_type<> ref() const noexcept;

 private:
  std::unique_ptr<impl_type> impl_;
};
}  // namespace cuco

#include <cuco/detail/distinct_count_estimator/distinct_count_estimator.inl>