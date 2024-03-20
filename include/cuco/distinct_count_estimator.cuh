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
#include <cuco/sketch_size.hpp>
#include <cuco/utility/allocator.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cstddef>
#include <iterator>
#include <memory>

namespace cuco {
/**
 * @brief A GPU-accelerated utility for approximating the number of distinct items in a multiset.
 *
 * @note This implementation is based on the HyperLogLog++ algorithm:
 * https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
 *
 * @tparam T Type of items to count
 * @tparam Scope The scope in which operations will be performed by individual threads
 * @tparam Hash Hash function used to hash items
 * @tparam Allocator Type of allocator used for device storage
 */
template <class T,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Hash               = cuco::xxhash_64<T>,
          class Allocator          = cuco::cuda_allocator<std::byte>>
class distinct_count_estimator {
  using impl_type = detail::hyperloglog<T, Scope, Hash, Allocator>;

 public:
  static constexpr auto thread_scope = impl_type::thread_scope;  ///< CUDA thread scope

  template <cuda::thread_scope NewScope = thread_scope>
  using ref_type = cuco::distinct_count_estimator_ref<T, NewScope, Hash>;  ///< Non-owning reference
                                                                           ///< type

  using value_type     = typename impl_type::value_type;      ///< Type of items to count
  using allocator_type = typename impl_type::allocator_type;  ///< Allocator type

  // TODO enable CTAD
  /**
   * @brief Constructs a `distinct_count_estimator` host object.
   *
   * @note This function synchronizes the given stream.
   *
   * @throw If sketch size < 0.0625KB or 64B
   * @throw If sketch storage has insufficient alignment
   *
   * @param sketch_size_kb Maximum sketch size in KB
   * @param hash The hash function used to hash items
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the object
   */
  constexpr distinct_count_estimator(cuco::sketch_size_kb sketch_size_kb = 32_KB,
                                     Hash const& hash                    = {},
                                     Allocator const& alloc              = {},
                                     cuco::cuda_stream_ref stream        = {});

  ~distinct_count_estimator() = default;

  distinct_count_estimator(distinct_count_estimator const&)            = delete;
  distinct_count_estimator& operator=(distinct_count_estimator const&) = delete;
  distinct_count_estimator(distinct_count_estimator&&) = default;  ///< Move constructor

  // TODO this is somehow required to pass the Doxygen check.
  /**
   * @brief Copy-assignment operator.
   *
   * @return Copy of `*this`
   */
  distinct_count_estimator& operator=(distinct_count_estimator&&) = default;

  /**
   * @brief Asynchronously resets the estimator, i.e., clears the current count estimate.
   *
   * @param stream CUDA stream this operation is executed in
   */
  void clear_async(cuco::cuda_stream_ref stream = {}) noexcept;

  /**
   * @brief Resets the estimator, i.e., clears the current count estimate.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `clear_async`.
   *
   * @param stream CUDA stream this operation is executed in
   */
  void clear(cuco::cuda_stream_ref stream = {});

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
  void add_async(InputIt first, InputIt last, cuco::cuda_stream_ref stream = {}) noexcept;

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
  void add(InputIt first, InputIt last, cuco::cuda_stream_ref stream = {});

  /**
   * @brief Asynchronously merges the result of `other` estimator into `*this` estimator.
   *
   * @throw If this->sketch_bytes() != other.sketch_bytes()
   *
   * @tparam OtherScope Thread scope of `other` estimator
   * @tparam OtherAllocator Allocator type of `other` estimator
   *
   * @param other Other estimator to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope, class OtherAllocator>
  void merge_async(distinct_count_estimator<T, OtherScope, Hash, OtherAllocator> const& other,
                   cuco::cuda_stream_ref stream = {});

  /**
   * @brief Merges the result of `other` estimator into `*this` estimator.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `merge_async`.
   *
   * @throw If this->sketch_bytes() != other.sketch_bytes()
   *
   * @tparam OtherScope Thread scope of `other` estimator
   * @tparam OtherAllocator Allocator type of `other` estimator
   *
   * @param other Other estimator to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope, class OtherAllocator>
  void merge(distinct_count_estimator<T, OtherScope, Hash, OtherAllocator> const& other,
             cuco::cuda_stream_ref stream = {});

  /**
   * @brief Asynchronously merges the result of `other` estimator reference into `*this` estimator.
   *
   * @throw If this->sketch_bytes() != other.sketch_bytes()
   *
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param other_ref Other estimator reference to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope>
  void merge_async(ref_type<OtherScope> const& other_ref, cuco::cuda_stream_ref stream = {});

  /**
   * @brief Merges the result of `other` estimator reference into `*this` estimator.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `merge_async`.
   *
   * @throw If this->sketch_bytes() != other.sketch_bytes()
   *
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param other_ref Other estimator reference to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope>
  void merge(ref_type<OtherScope> const& other_ref, cuco::cuda_stream_ref stream = {});

  /**
   * @brief Compute the estimated distinct items count.
   *
   * @note This function synchronizes the given stream.
   *
   * @param stream CUDA stream this operation is executed in
   *
   * @return Approximate distinct items count
   */
  [[nodiscard]] std::size_t estimate(cuco::cuda_stream_ref stream = {}) const;

  /**
   * @brief Get device ref.
   *
   * @return Device ref object of the current `distinct_count_estimator` host object
   */
  [[nodiscard]] ref_type<> ref() const noexcept;

  /**
   * @brief Get hash function.
   *
   * @return The hash function
   */
  [[nodiscard]] auto hash() const noexcept;

  /**
   * @brief Gets the span of the sketch.
   *
   * @return The cuda::std::span of the sketch
   */
  [[nodiscard]] cuda::std::span<std::byte> sketch() const noexcept;

  /**
   * @brief Gets the number of bytes required for the sketch storage.
   *
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] constexpr std::size_t sketch_bytes() const noexcept;

  /**
   * @brief Gets the number of bytes required for the sketch storage.
   *
   * @param sketch_size_kb Upper bound sketch size in KB
   *
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] static constexpr std::size_t sketch_bytes(
    cuco::sketch_size_kb sketch_size_kb) noexcept;

  /**
   * @brief Gets the alignment required for the sketch storage.
   *
   * @return The required alignment
   */
  [[nodiscard]] static constexpr std::size_t sketch_alignment() noexcept;

 private:
  std::unique_ptr<impl_type> impl_;  ///< Implementation object
};
}  // namespace cuco

#include <cuco/detail/distinct_count_estimator/distinct_count_estimator.inl>