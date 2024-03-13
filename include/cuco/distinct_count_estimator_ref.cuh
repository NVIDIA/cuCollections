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
#include <cuco/detail/hyperloglog/hyperloglog_ref.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cooperative_groups.h>

#include <cstddef>

namespace cuco {
/**
 * @brief A GPU-accelerated utility for approximating the number of distinct items in a multiset.
 *
 * @note This implementation is based on the HyperLogLog++ algorithm:
 * https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
 * @note The `Precision` parameter can be used to trade runtime/memory footprint for better
 * accuracy. A higher value corresponds to a more accurate result, however, setting the precision
 * too high will result in deminishing results.
 *
 * @tparam T Type of items to count
 * @tparam Precision Tuning parameter to trade runtime/memory footprint for better accuracy
 * @tparam Scope The scope in which operations will be performed by individual threads
 * @tparam Hash Hash function used to hash items
 */
template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash>
class distinct_count_estimator_ref {
  using impl_type = detail::hyperloglog_ref<T, Precision, Scope, Hash>;

 public:
  static constexpr auto thread_scope = impl_type::thread_scope;  ///< CUDA thread scope
  static constexpr auto precision    = impl_type::precision;     ///< Precision

  using value_type = typename impl_type::value_type;  ///< Type of items to count

  template <cuda::thread_scope NewScope>
  using with_scope =
    distinct_count_estimator_ref<T, Precision, NewScope, Hash>;  ///< Ref type with different thread
                                                                 ///< scope

  // TODO let storage_type be inferred?
  /**
   * @brief Constructs a non-owning `distinct_count_estimator_ref` object.
   *
   * @param sketch_span Reference to sketch storage
   * @param hash The hash function used to hash items
   */
  template <class U, std::size_t N>
  __host__ __device__ constexpr distinct_count_estimator_ref(cuda::std::span<U, N> sketch_span,
                                                             Hash const& hash = {}) noexcept;

  /**
   * @brief Resets the estimator, i.e., clears the current count estimate.
   *
   * @tparam CG CUDA Cooperative Group type
   *
   * @param group CUDA Cooperative group this operation is executed in
   */
  template <class CG>
  __device__ void clear(CG const& group) noexcept;

  /**
   * @brief Asynchronously resets the estimator, i.e., clears the current count estimate.
   *
   * @param stream CUDA stream this operation is executed in
   */
  __host__ void clear_async(cuco::cuda_stream_ref stream = {}) noexcept;

  /**
   * @brief Resets the estimator, i.e., clears the current count estimate.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `clear_async`.
   *
   * @param stream CUDA stream this operation is executed in
   */
  __host__ void clear(cuco::cuda_stream_ref stream = {});

  /**
   * @brief Adds an item to the estimator.
   *
   * @param item The item to be counted
   */
  __device__ void add(T const& item) noexcept;

  /**
   * @brief Asynchronously adds to be counted items to the estimator.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * T></tt> is `true`
   *
   * @param first Beginning of the sequence of items
   * @param last End of the sequence of items
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt>
  __host__ void add_async(InputIt first, InputIt last, cuco::cuda_stream_ref stream = {});

  /**
   * @brief Adds to be counted items to the estimator.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `add_async`.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * T></tt> is `true`
   *
   * @param first Beginning of the sequence of items
   * @param last End of the sequence of items
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt>
  __host__ void add(InputIt first, InputIt last, cuco::cuda_stream_ref stream = {});

  /**
   * @brief Merges the result of `other` estimator reference into `*this` estimator reference.
   *
   * @tparam CG CUDA Cooperative Group type
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param group CUDA Cooperative group this operation is executed in
   * @param other Other estimator reference to be merged into `*this`
   */
  template <class CG, cuda::thread_scope OtherScope>
  __device__ void merge(
    CG const& group,
    distinct_count_estimator_ref<T, Precision, OtherScope, Hash> const& other) noexcept;

  /**
   * @brief Asynchronously merges the result of `other` estimator reference into `*this` estimator.
   *
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param other Other estimator reference to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope>
  __host__ void merge_async(
    distinct_count_estimator_ref<T, Precision, OtherScope, Hash> const& other,
    cuco::cuda_stream_ref stream = {}) noexcept;

  /**
   * @brief Merges the result of `other` estimator reference into `*this` estimator.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `merge_async`.
   *
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param other Other estimator reference to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope>
  __host__ void merge(distinct_count_estimator_ref<T, Precision, OtherScope, Hash> const& other,
                      cuco::cuda_stream_ref stream = {});

  /**
   * @brief Compute the estimated distinct items count.
   *
   * @param group CUDA thread block group this operation is executed in
   *
   * @return Approximate distinct items count
   */
  [[nodiscard]] __device__ std::size_t estimate(
    cooperative_groups::thread_block const& group) const noexcept;

  /**
   * @brief Compute the estimated distinct items count.
   *
   * @note This function synchronizes the given stream.
   *
   * @param stream CUDA stream this operation is executed in
   *
   * @return Approximate distinct items count
   */
  [[nodiscard]] __host__ std::size_t estimate(cuco::cuda_stream_ref stream = {}) const;

  /**
   * @brief Gets the hash function.
   *
   * @return The hash function
   */
  [[nodiscard]] __host__ __device__ auto hash() const noexcept;

  /**
   * @brief Gets the span of the sketch.
   *
   * @return The cuda::std::span of the sketch
   */
  [[nodiscard]] __host__ __device__ auto sketch() const noexcept;

  /**
   * @brief Gets the number of bytes required for the sketch storage.
   *
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] __host__ __device__ static constexpr std::size_t sketch_bytes() noexcept;

  /**
   * @brief Gets the alignment required for the sketch storage.
   *
   * @return The required alignment
   */
  [[nodiscard]] __host__ __device__ static constexpr std::size_t sketch_alignment() noexcept;

 private:
  impl_type impl_;  ///< Implementation object

  template <class T_, int32_t Precision_, cuda::thread_scope Scope_, class Hash_>
  friend class distinct_count_estimator_ref;
};
}  // namespace cuco

#include <cuco/detail/distinct_count_estimator/distinct_count_estimator_ref.inl>