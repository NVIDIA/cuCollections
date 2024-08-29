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

/**
 * @brief A GPU-accelerated Blocked Bloom filter.
 *
 * The `bloom_filter` supports two types of operations:
 * - Host-side "bulk" operations
 * - Device-side "singular" operations
 *
 * The host-side bulk operations include `add`, `test`, etc. These APIs should be used when
 * there are a large number of keys to modify or lookup. For example, given a range of keys
 * specified by device-accessible iterators, the bulk `add` function will insert all keys into
 * the filter.
 *
 * The singular device-side operations allow individual threads (or Cooperative Groups) to perform
 * independent modify or lookup operations from device code. These operations are accessed through
 * non-owning, trivially copyable reference types (or "ref").
 *
 * @tparam Key Key type
 * @tparam Extent Size type that is used to determine the number of blocks in the filter
 * @tparam Scope The scope in which operations will be performed by individual threads
 * @tparam Hash Hash function used to generate a key's fingerprint
 * @tparam Allocator Type of allocator used for device-accessible storage
 * @tparam BlockWords Number of machine words in a filter block
 * @tparam Word Underlying machine word type that can be updated atomically
 */
template <class Key,
          class Extent             = cuco::extent<std::size_t>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Hash               = cuco::xxhash_64<Key>,
          class Allocator          = cuco::cuda_allocator<std::byte>,
          std::uint32_t BlockWords = 4,
          class Word               = std::uint64_t>
class bloom_filter {
 public:
  /**
   * @brief Non-owning filter ref type
   *
   * @tparam NewScope Thead scope of the ref type
   */
  template <cuda::thread_scope NewScope = Scope>
  using ref_type = bloom_filter_ref<Key, Extent, NewScope, Hash, BlockWords, Word>;

  static constexpr auto thread_scope = ref_type<>::thread_scope;  ///< CUDA thread scope
  static constexpr auto block_words =
    ref_type<>::block_words;  ///< Number of machine words in each filter block

  using key_type    = typename ref_type<>::key_type;     ///< Key Type
  using extent_type = typename ref_type<>::extent_type;  ///< Extent type
  using size_type   = typename extent_type::value_type;  ///< Underlying type of the extent type
  using hasher      = typename ref_type<>::hasher;       ///< Hash function type
  using word_type   = typename ref_type<>::word_type;    ///< Machine word type
  using allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<word_type>;  ///< Allocator
                                                                                  ///< type

  bloom_filter(bloom_filter const&) = delete;  ///< Copy constructor is not available
  bloom_filter& operator=(bloom_filter const&) =
    delete;  ///< Copy-assignment constructor is not available

  bloom_filter(bloom_filter&&)            = default;  ///< Move constructor
  bloom_filter& operator=(bloom_filter&&) = default;  ///< Move-assignment constructor

  ~bloom_filter() = default;  ///< Destructor

  /**
   * @brief Constructs a statically-sized Bloom filter.
   *
   * @note The total number of bits in the filter is determined by `BlockWords * num_blocks`.
   *
   * @param num_blocks Number of sub-filters or blocks
   * @param pattern_bits Number of bits in a key's fingerprint
   * @param hash Hash function used to generate a key's fingerprint
   * @param alloc Allocator used for allocating device-accessible storage
   * @param stream CUDA stream used to initialize the filter
   */
  __host__ bloom_filter(Extent num_blocks,
                        std::uint32_t pattern_bits,
                        cuda_thread_scope<Scope> = {},
                        Hash const& hash         = {},
                        Allocator const& alloc   = {},
                        cuda::stream_ref stream  = {});

  /**
   * @brief Erases all information from the filter.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `clear_async`.
   *
   * @param stream CUDA stream this operation is executed in
   */
  __host__ void clear(cuda::stream_ref stream = {});

  /**
   * @brief Asynchronously erases all information from the filter.
   *
   * @param stream CUDA stream this operation is executed in
   */
  __host__ void clear_async(cuda::stream_ref stream = {});

  /**
   * @brief Adds all keys in the range `[first, last)` to the filter.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `add_async`.
   *
   * @tparam InputIt Device-accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * bloom_filter<K>::key_type></tt> is `true`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt>
  __host__ void add(InputIt first, InputIt last, cuda::stream_ref stream = {});

  /**
   * @brief Asynchrounously adds all keys in the range `[first, last)` to the filter.
   *
   * @tparam InputIt Device-accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * bloom_filter<K>::key_type></tt> is `true`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt>
  __host__ void add_async(InputIt first, InputIt last, cuda::stream_ref stream = {});

  /**
   * @brief Adds keys in the range `[first, last)` if `pred` of the corresponding `stencil` returns
   * `true`.
   *
   * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns `true`.
   * @note This function synchronizes the given stream and returns the number of successful
   * insertions. For asynchronous execution use `add_if_async`.
   *
   * @tparam InputIt Device-accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * bloom_filter<K>::key_type></tt> is `true`
   * @tparam StencilIt Device-accessible random-access iterator whose `value_type` is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt, class StencilIt, class Predicate>
  __host__ void add_if(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream = {});

  /**
   * @brief Asynchronously adds keys in the range `[first, last)` if `pred` of the corresponding
   * `stencil` returns `true`.
   *
   * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns `true`.
   *
   * @tparam InputIt Device-accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * bloom_filter<K>::key_type></tt> is `true`
   * @tparam StencilIt Device-accessible random-access iterator whose `value_type` is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt, class StencilIt, class Predicate>
  __host__ void add_if_async(InputIt first,
                             InputIt last,
                             StencilIt stencil,
                             Predicate pred,
                             cuda::stream_ref stream = {}) noexcept;

  /**
   * @brief Tests all keys in the range `[first, last)` if their fingerprints are present in the
   * filter.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `test_async`.
   *
   * @tparam InputIt Device-accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * bloom_filter<K>::key_type></tt> is `true`
   * @tparam OutputIt Device-accessible output iterator assignable from `bool`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt, class OutputIt>
  __host__ void test(InputIt first,
                     InputIt last,
                     OutputIt output_begin,
                     cuda::stream_ref stream = {}) const;

  /**
   * @brief Asynchronously tests all keys in the range `[first, last)` if their fingerprints are
   * present in the filter.
   *
   * @tparam InputIt Device-accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * bloom_filter<K>::key_type></tt> is `true`
   * @tparam OutputIt Device-accessible output iterator assignable from `bool`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt, class OutputIt>
  __host__ void test_async(InputIt first,
                           InputIt last,
                           OutputIt output_begin,
                           cuda::stream_ref stream = {}) const noexcept;

  /**
   * @brief Tests all keys in the range `[first, last)` if their fingerprints are present in the
   * filter if `pred` of the corresponding `stencil` returns `true`.
   *
   * @note The key `*(first + i)` is queried if `pred( *(stencil + i) )` returns `true`.
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `test_if_async`.
   *
   * @tparam InputIt Device-accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * bloom_filter<K>::key_type></tt> is `true`
   * @tparam StencilIt Device-accessible random-access iterator whose `value_type` is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   * @tparam OutputIt Device-accessible output iterator assignable from `bool`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void test_if(InputIt first,
                        InputIt last,
                        StencilIt stencil,
                        Predicate pred,
                        OutputIt output_begin,
                        cuda::stream_ref stream = {}) const;

  /**
   * @brief Asynchronously tests all keys in the range `[first, last)` if their fingerprints are
   * present in the filter if `pred` of the corresponding `stencil` returns `true`.
   *
   * @note The key `*(first + i)` is queried if `pred( *(stencil + i) )` returns `true`.
   *
   * @tparam InputIt Device-accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * bloom_filter<K>::key_type></tt> is `true`
   * @tparam StencilIt Device-accessible random-access iterator whose `value_type` is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   * @tparam OutputIt Device-accessible output iterator assignable from `bool`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void test_if_async(InputIt first,
                              InputIt last,
                              StencilIt stencil,
                              Predicate pred,
                              OutputIt output_begin,
                              cuda::stream_ref stream = {}) const noexcept;

  /**
   * @brief Gets a pointer to the underlying filter storage.
   *
   * @return Pointer to the underlying filter storage
   */
  [[nodiscard]] __host__ word_type* data() noexcept;

  /**
   * @brief Gets a pointer to the underlying filter storage.
   *
   * @return Pointer to the underlying filter storage
   */
  [[nodiscard]] __host__ word_type const* data() const noexcept;

  /**
   * @brief Gets the number of sub-filter blocks.
   *
   * @return Number of sub-filter blocks
   */
  [[nodiscard]] __host__ extent_type block_extent() const noexcept;

  /**
   * @brief Gets the function used to hash keys
   *
   * @return The function used to hash keys
   */
  [[nodiscard]] __host__ hasher hash_function() const noexcept;

  /**
   * @brief Gets the allocator.
   *
   * @return The allocator
   */
  [[nodiscard]] __host__ allocator_type allocator() const noexcept;

  /**
   * @brief Get device ref.
   *
   * @return Device ref of the current `bloom_filter` object
   */
  [[nodiscard]] __host__ ref_type<> ref() const noexcept;

 private:
  allocator_type allocator_;  ///< Allocator used to allocate device-accessible storage
  std::unique_ptr<word_type, detail::custom_deleter<std::size_t, allocator_type>>
    data_;          ///< Storage of the current `bloom_filter` object
  ref_type<> ref_;  ///< Device ref of the current `bloom_filter` object
};

}  // namespace cuco

#include <cuco/detail/bloom_filter/bloom_filter.inl>