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

#include <cuco/detail/error.hpp>
#include <cuco/detail/hyperloglog/hyperloglog_ref.cuh>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/types.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/stream_ref>

#include <cstddef>
#include <iterator>
#include <memory>

namespace cuco::detail {
/**
 * @brief A GPU-accelerated utility for approximating the number of distinct items in a multiset.
 *
 * @note This class implements the HyperLogLog/HyperLogLog++ algorithm:
 * https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
 *
 * @tparam T Type of items to count
 * @tparam Scope The scope in which operations will be performed by individual threads
 * @tparam Hash Hash function used to hash items
 * @tparam Allocator Type of allocator used for device storage
 */
template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
class hyperloglog {
 public:
  static constexpr auto thread_scope = Scope;  ///< CUDA thread scope

  template <cuda::thread_scope NewScope = thread_scope>
  using ref_type = hyperloglog_ref<T, NewScope, Hash>;  ///< Non-owning reference
                                                        ///< type

  using value_type    = typename ref_type<>::value_type;     ///< Type of items to count
  using hasher        = typename ref_type<>::hasher;         ///< Hash function type
  using register_type = typename ref_type<>::register_type;  ///< HLL register type
  using allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<register_type>;  ///< Allocator
                                                                                      ///< type

 private:
  /**
   * @brief Constructs a `hyperloglog` host object.
   *
   * @note This function synchronizes the given stream.
   *
   * @param sketch_size_b Sketch size in bytes
   * @param hash The hash function used to hash items
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the object
   */
  constexpr hyperloglog(std::size_t sketch_size_b,
                        Hash const& hash,
                        Allocator const& alloc,
                        cuda::stream_ref stream)
    : allocator_{alloc},
      sketch_{this->allocator_.allocate(sketch_size_b / sizeof(register_type)),
              custom_deleter{sketch_size_b / sizeof(register_type), this->allocator_}},
      ref_{cuda::std::span{reinterpret_cast<std::byte*>(this->sketch_.get()), sketch_size_b}, hash}
  {
    this->ref_.clear_async(stream);
  }

 public:
  /**
   * @brief Constructs a `hyperloglog` host object.
   *
   * @note This function synchronizes the given stream.
   *
   * @param sketch_size_kb Maximum sketch size in KB
   * @param hash The hash function used to hash items
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the object
   */
  constexpr hyperloglog(cuco::sketch_size_kb sketch_size_kb,
                        Hash const& hash,
                        Allocator const& alloc,
                        cuda::stream_ref stream)
    : hyperloglog{sketch_bytes(sketch_size_kb), hash, alloc, stream}
  {
  }

  /**
   * @brief Constructs a `hyperloglog` host object.
   *
   * @note This function synchronizes the given stream.
   *
   * @param standard_deviation Desired standard deviation for the approximation error
   * @param hash The hash function used to hash items
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the object
   */
  constexpr hyperloglog(cuco::standard_deviation standard_deviation,
                        Hash const& hash,
                        Allocator const& alloc,
                        cuda::stream_ref stream)
    : hyperloglog{sketch_bytes(standard_deviation), hash, alloc, stream}
  {
  }

  ~hyperloglog() = default;

  hyperloglog(hyperloglog const&)            = delete;
  hyperloglog& operator=(hyperloglog const&) = delete;
  hyperloglog(hyperloglog&&)                 = default;  ///< Move constructor

  /**
   * @brief Copy-assignment operator.
   *
   * @return Copy of `*this`
   */
  hyperloglog& operator=(hyperloglog&&) = default;

  /**
   * @brief Asynchronously resets the estimator, i.e., clears the current count estimate.
   *
   * @param stream CUDA stream this operation is executed in
   */
  constexpr void clear_async(cuda::stream_ref stream) noexcept { this->ref_.clear_async(stream); }

  /**
   * @brief Resets the estimator, i.e., clears the current count estimate.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `clear_async`.
   *
   * @param stream CUDA stream this operation is executed in
   */
  constexpr void clear(cuda::stream_ref stream) { this->ref_.clear(stream); }

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
  constexpr void add_async(InputIt first, InputIt last, cuda::stream_ref stream)
  {
    this->ref_.add_async(first, last, stream);
  }

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
  constexpr void add(InputIt first, InputIt last, cuda::stream_ref stream)
  {
    this->ref_.add(first, last, stream);
  }

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
  constexpr void merge_async(hyperloglog<T, OtherScope, Hash, OtherAllocator> const& other,
                             cuda::stream_ref stream)
  {
    this->ref_.merge_async(other.ref(), stream);
  }

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
  constexpr void merge(hyperloglog<T, OtherScope, Hash, OtherAllocator> const& other,
                       cuda::stream_ref stream)
  {
    this->ref_.merge(other.ref(), stream);
  }

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
  constexpr void merge_async(ref_type<OtherScope> const& other_ref, cuda::stream_ref stream)
  {
    this->ref_.merge_async(other_ref, stream);
  }

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
  constexpr void merge(ref_type<OtherScope> const& other_ref, cuda::stream_ref stream)
  {
    this->ref_.merge(other_ref, stream);
  }

  /**
   * @brief Compute the estimated distinct items count.
   *
   * @note This function synchronizes the given stream.
   *
   * @param stream CUDA stream this operation is executed in
   *
   * @return Approximate distinct items count
   */
  [[nodiscard]] constexpr std::size_t estimate(cuda::stream_ref stream) const
  {
    return this->ref_.estimate(stream);
  }

  /**
   * @brief Get device ref.
   *
   * @return Device ref object of the current `distinct_count_estimator` host object
   */
  [[nodiscard]] constexpr ref_type<> ref() const noexcept { return this->ref_; }

  /**
   * @brief Get hash function.
   *
   * @return The hash function
   */
  [[nodiscard]] constexpr auto hash_function() const noexcept { return this->ref_.hash_function(); }

  /**
   * @brief Gets the span of the sketch.
   *
   * @return The cuda::std::span of the sketch
   */
  [[nodiscard]] constexpr cuda::std::span<std::byte> sketch() const noexcept
  {
    return this->ref_.sketch();
  }

  /**
   * @brief Gets the number of bytes required for the sketch storage.
   *
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] constexpr std::size_t sketch_bytes() const noexcept
  {
    return this->ref_.sketch_bytes();
  }

  /**
   * @brief Gets the number of bytes required for the sketch storage.
   *
   * @param sketch_size_kb Upper bound sketch size in KB
   *
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] static constexpr std::size_t sketch_bytes(
    cuco::sketch_size_kb sketch_size_kb) noexcept
  {
    return ref_type<>::sketch_bytes(sketch_size_kb);
  }

  /**
   * @brief Gets the number of bytes required for the sketch storage.
   *
   * @param standard_deviation Upper bound standard deviation for approximation error
   *
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] static constexpr std::size_t sketch_bytes(
    cuco::standard_deviation standard_deviation) noexcept
  {
    return ref_type<>::sketch_bytes(standard_deviation);
  }

  /**
   * @brief Gets the alignment required for the sketch storage.
   *
   * @return The required alignment
   */
  [[nodiscard]] static constexpr std::size_t sketch_alignment() noexcept
  {
    return ref_type<>::sketch_alignment();
  }

 private:
  allocator_type allocator_;  ///< Storage allocator
  std::unique_ptr<register_type, custom_deleter<std::size_t, allocator_type>>
    sketch_;        ///< Sketch storage
  ref_type<> ref_;  //< Ref type

  // Needs to be friends with other instantiations of this class template to have access to their
  // storage
  template <class T_, cuda::thread_scope Scope_, class Hash_, class Allocator_>
  friend class hyperloglog;
};
}  // namespace cuco::detail