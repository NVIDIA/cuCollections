/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuco/detail/__config>
#include <cuco/detail/common_functors.cuh>
#include <cuco/detail/common_kernels.cuh>
#include <cuco/detail/storage/counter_storage.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/extent.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/storage.cuh>
#include <cuco/utility/traits.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_select.cuh>

#include <cuda/atomic>

namespace cuco {
namespace experimental {
namespace detail {
/**
 * @brief An open addressing impl class.
 *
 * @note This class should NOT be used directly.
 *
 * @throw If the size of the given key type is larger than 8 bytes
 * @throw If the size of the given slot type is larger than 16 bytes
 * @throw If the given key type doesn't have unique object representations, i.e.,
 * `cuco::bitwise_comparable_v<Key> == false`
 * @throw If the probing scheme type is not inherited from `cuco::detail::probing_scheme_base`
 *
 * @tparam Key Type used for keys. Requires `cuco::is_bitwise_comparable_v<Key>`
 * @tparam Value Type used for storage values.
 * @tparam Extent Data structure size type
 * @tparam Scope The scope in which operations will be performed by individual threads.
 * @tparam KeyEqual Binary callable type used to compare two keys for equality
 * @tparam ProbingScheme Probing scheme (see `include/cuco/probing_scheme.cuh` for choices)
 * @tparam Allocator Type of allocator used for device storage
 * @tparam Storage Slot window storage type
 */
template <class Key,
          class Value,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
class open_addressing_impl {
  static_assert(sizeof(Key) <= 8, "Container does not support key types larger than 8 bytes.");

  static_assert(sizeof(Value) <= 16, "Container does not support slot types larger than 16 bytes.");

  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

  static_assert(
    std::is_base_of_v<cuco::experimental::detail::probing_scheme_base<ProbingScheme::cg_size>,
                      ProbingScheme>,
    "ProbingScheme must inherit from cuco::detail::probing_scheme_base");

 public:
  static constexpr auto cg_size      = ProbingScheme::cg_size;  ///< CG size used for probing
  static constexpr auto window_size  = Storage::window_size;    ///< Window size used for probing
  static constexpr auto thread_scope = Scope;                   ///< CUDA thread scope

  using key_type   = Key;    ///< Key type
  using value_type = Value;  ///< The storage value type, NOT payload type
  /// Extent type
  using extent_type = decltype(make_window_extent<open_addressing_impl>(std::declval<Extent>()));
  using size_type   = typename extent_type::value_type;  ///< Size type
  using key_equal   = KeyEqual;                          ///< Key equality comparator type
  using storage_type =
    detail::storage<Storage, value_type, extent_type, Allocator>;  ///< Storage type
  using allocator_type = typename storage_type::allocator_type;    ///< Allocator type

  using storage_ref_type = typename storage_type::ref_type;  ///< Non-owning window storage ref type
  using probing_scheme_type = ProbingScheme;                 ///< Probe scheme type

  /**
   * @brief Constructs a statically-sized open addressing data structure with the specified initial
   * capacity, sentinel values and CUDA stream.
   *
   * @note The actual capacity depends on the given `capacity`, the probing scheme, CG size, and the
   * window size and it is computed via the `make_window_extent` factory. Insert operations will not
   * automatically grow the container. Attempting to insert more unique keys than the capacity of
   * the container results in undefined behavior.
   * @note Any `*_sentinel`s are reserved and behavior is undefined when attempting to insert
   * this sentinel value.
   * @note If a non-default CUDA stream is provided, the caller is responsible for synchronizing the
   * stream before the object is first used.
   *
   * @param capacity The requested lower-bound size
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param empty_slot_sentinel The reserved slot value for empty slots
   * @param pred Key equality binary predicate
   * @param probing_scheme Probing scheme
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the data structure
   */
  constexpr open_addressing_impl(Extent capacity,
                                 key_type empty_key_sentinel,
                                 value_type empty_slot_sentinel,
                                 KeyEqual const& pred,
                                 ProbingScheme const& probing_scheme,
                                 Allocator const& alloc,
                                 cuda_stream_ref stream) noexcept
    : empty_key_sentinel_{empty_key_sentinel},
      empty_slot_sentinel_{empty_slot_sentinel},
      predicate_{pred},
      probing_scheme_{probing_scheme},
      storage_{make_window_extent<open_addressing_impl>(capacity), alloc}
  {
    this->clear_async(stream);
  }

  /**
   * @brief Erases all elements from the container. After this call, `size()` returns zero.
   * Invalidates any references, pointers, or iterators referring to contained elements.
   *
   * @param stream CUDA stream this operation is executed in
   */
  void clear(cuda_stream_ref stream) noexcept
  {
    this->clear_async(stream);
    stream.synchronize();
  }

  /**
   * @brief Asynchronously erases all elements from the container. After this call, `size()` returns
   * zero. Invalidates any references, pointers, or iterators referring to contained elements.
   *
   * @param stream CUDA stream this operation is executed in
   */
  void clear_async(cuda_stream_ref stream) noexcept
  {
    storage_.initialize(empty_slot_sentinel_, stream);
  }

  /**
   * @brief Inserts all keys in the range `[first, last)` and returns the number of successful
   * insertions.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `insert_async`.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * open_addressing_impl::value_type></tt> is `true`
   * @tparam Ref Type of non-owning device container ref allowing access to storage
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param container_ref Non-owning device container ref used to access the slot storage
   * @param stream CUDA stream used for insert
   *
   * @return Number of successfully inserted keys
   */
  template <typename InputIt, typename Ref>
  size_type insert(InputIt first, InputIt last, Ref container_ref, cuda_stream_ref stream)
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return 0; }

    auto counter =
      detail::counter_storage<size_type, thread_scope, allocator_type>{this->allocator()};
    counter.reset(stream);

    auto const grid_size = cuco::detail::grid_size(num_keys, cg_size);

    auto const always_true = thrust::constant_iterator<bool>{true};
    detail::insert_if_n<cg_size, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
        first, num_keys, always_true, thrust::identity{}, counter.data(), container_ref);

    return counter.load_to_host(stream);
  }

  /**
   * @brief Asynchronously inserts all keys in the range `[first, last)`.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * open_addressing_impl::value_type></tt> is `true`
   * @tparam Ref Type of non-owning device container ref allowing access to storage
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param container_ref Non-owning device container ref used to access the slot storage
   * @param stream CUDA stream used for insert
   */
  template <typename InputIt, typename Ref>
  void insert_async(InputIt first, InputIt last, Ref container_ref, cuda_stream_ref stream) noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto const grid_size = cuco::detail::grid_size(num_keys, cg_size);

    auto const always_true = thrust::constant_iterator<bool>{true};
    detail::insert_if_n<cg_size, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
        first, num_keys, always_true, thrust::identity{}, container_ref);
  }

  /**
   * @brief Inserts keys in the range `[first, last)` if `pred` of the corresponding stencil returns
   * true.
   *
   * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns true.
   * @note This function synchronizes the given stream and returns the number of successful
   * insertions. For asynchronous execution use `insert_if_async`.
   *
   * @tparam InputIt Device accessible random access iterator whose `value_type` is
   * convertible to the container's `value_type`
   * @tparam StencilIt Device accessible random access iterator whose value_type is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   * @tparam Ref Type of non-owning device container ref allowing access to storage
   *
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param container_ref Non-owning device container ref used to access the slot storage
   * @param stream CUDA stream used for the operation
   *
   * @return Number of successfully inserted keys
   */
  template <typename InputIt, typename StencilIt, typename Predicate, typename Ref>
  size_type insert_if(InputIt first,
                      InputIt last,
                      StencilIt stencil,
                      Predicate pred,
                      Ref container_ref,
                      cuda_stream_ref stream)
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return 0; }

    auto counter =
      detail::counter_storage<size_type, thread_scope, allocator_type>{this->allocator()};
    counter.reset(stream);

    auto const grid_size = cuco::detail::grid_size(num_keys, cg_size);

    detail::insert_if_n<cg_size, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
        first, num_keys, stencil, pred, counter.data(), container_ref);

    return counter.load_to_host(stream);
  }

  /**
   * @brief Asynchronously inserts keys in the range `[first, last)` if `pred` of the corresponding
   * stencil returns true.
   *
   * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns true.
   *
   * @tparam InputIt Device accessible random access iterator whose `value_type` is
   * convertible to the container's `value_type`
   * @tparam StencilIt Device accessible random access iterator whose value_type is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   * @tparam Ref Type of non-owning device container ref allowing access to storage
   *
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param container_ref Non-owning device container ref used to access the slot storage
   * @param stream CUDA stream used for the operation
   */
  template <typename InputIt, typename StencilIt, typename Predicate, typename Ref>
  void insert_if_async(InputIt first,
                       InputIt last,
                       StencilIt stencil,
                       Predicate pred,
                       Ref container_ref,
                       cuda_stream_ref stream) noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto const grid_size = cuco::detail::grid_size(num_keys, cg_size);

    detail::insert_if_n<cg_size, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
        first, num_keys, stencil, pred, container_ref);
  }

  /**
   * @brief Asynchronously indicates whether the keys in the range `[first, last)` are contained in
   * the container.
   *
   * @tparam InputIt Device accessible input iterator
   * @tparam OutputIt Device accessible output iterator assignable from `bool`
   * @tparam Ref Type of non-owning device container ref allowing access to storage
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param container_ref Non-owning device container ref used to access the slot storage
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt, typename OutputIt, typename Ref>
  void contains_async(InputIt first,
                      InputIt last,
                      OutputIt output_begin,
                      Ref container_ref,
                      cuda_stream_ref stream) const noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto const grid_size = cuco::detail::grid_size(num_keys, cg_size);

    auto const always_true = thrust::constant_iterator<bool>{true};
    detail::contains_if_n<cg_size, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
        first, num_keys, always_true, thrust::identity{}, output_begin, container_ref);
  }

  /**
   * @brief Asynchronously indicates whether the keys in the range `[first, last)` are contained in
   * the container if `pred` of the corresponding stencil returns true.
   *
   * @note If `pred( *(stencil + i) )` is true, stores `true` or `false` to `(output_begin + i)`
   * indicating if the key `*(first + i)` is present int the container. If `pred( *(stencil + i) )`
   * is false, stores false to `(output_begin + i)`.
   *
   * @tparam InputIt Device accessible input iterator
   * @tparam StencilIt Device accessible random access iterator whose value_type is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   * @tparam OutputIt Device accessible output iterator assignable from `bool`
   * @tparam Ref Type of non-owning device container ref allowing access to storage
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param container_ref Non-owning device container ref used to access the slot storage
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt,
            typename StencilIt,
            typename Predicate,
            typename OutputIt,
            typename Ref>
  void contains_if_async(InputIt first,
                         InputIt last,
                         StencilIt stencil,
                         Predicate pred,
                         OutputIt output_begin,
                         Ref container_ref,
                         cuda_stream_ref stream) const noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto const grid_size = cuco::detail::grid_size(num_keys, cg_size);

    detail::contains_if_n<cg_size, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
        first, num_keys, stencil, pred, output_begin, container_ref);
  }

  /**
   * @brief Retrieves all keys contained in the container.
   *
   * @note This API synchronizes the given stream.
   * @note The order in which keys are returned is implementation defined and not guaranteed to be
   * consistent between subsequent calls to `retrieve_all`.
   * @note Behavior is undefined if the range beginning at `output_begin` is smaller than the return
   * value of `size()`.
   *
   * @tparam InputIt Device accessible container slot iterator
   * @tparam OutputIt Device accessible random access output iterator whose `value_type` is
   * convertible from the container's `value_type`
   * @tparam Predicate Type of predicate indicating if the given slot is filled
   *
   * @param begin Beginning of the container slot iterator
   * @param output_begin Beginning output iterator for keys
   * @param is_filled Predicate indicating if the given slot is filled
   * @param stream CUDA stream used for this operation
   *
   * @return Iterator indicating the end of the output
   */
  template <typename InputIt, typename OutputIt, typename Predicate>
  [[nodiscard]] OutputIt retrieve_all(InputIt begin,
                                      OutputIt output_begin,
                                      Predicate const& is_filled,
                                      cuda_stream_ref stream) const
  {
    std::size_t temp_storage_bytes = 0;
    using temp_allocator_type = typename std::allocator_traits<allocator_type>::rebind_alloc<char>;
    auto temp_allocator       = temp_allocator_type{this->allocator()};
    auto d_num_out            = reinterpret_cast<size_type*>(
      std::allocator_traits<temp_allocator_type>::allocate(temp_allocator, sizeof(size_type)));
    CUCO_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                        temp_storage_bytes,
                                        begin,
                                        output_begin,
                                        d_num_out,
                                        this->capacity(),
                                        is_filled,
                                        stream));

    // Allocate temporary storage
    auto d_temp_storage = temp_allocator.allocate(temp_storage_bytes);

    CUCO_CUDA_TRY(cub::DeviceSelect::If(d_temp_storage,
                                        temp_storage_bytes,
                                        begin,
                                        output_begin,
                                        d_num_out,
                                        this->capacity(),
                                        is_filled,
                                        stream));

    size_type h_num_out;
    CUCO_CUDA_TRY(
      cudaMemcpyAsync(&h_num_out, d_num_out, sizeof(size_type), cudaMemcpyDeviceToHost, stream));
    stream.synchronize();
    std::allocator_traits<temp_allocator_type>::deallocate(
      temp_allocator, reinterpret_cast<char*>(d_num_out), sizeof(size_type));
    temp_allocator.deallocate(d_temp_storage, temp_storage_bytes);

    return output_begin + h_num_out;
  }

  /**
   * @brief Gets the number of elements in the container.
   *
   * @note This function synchronizes the given stream.
   *
   * @tparam Predicate Type of predicate indicating if the given slot is filled
   *
   * @param is_filled Predicate indicating if the given slot is filled
   * @param stream CUDA stream used to get the number of inserted elements
   *
   * @return The number of elements in the container
   */
  template <typename Predicate>
  [[nodiscard]] size_type size(Predicate const& is_filled, cuda_stream_ref stream) const noexcept
  {
    auto counter =
      detail::counter_storage<size_type, thread_scope, allocator_type>{this->allocator()};
    counter.reset(stream);

    auto const grid_size = cuco::detail::grid_size(storage_.num_windows());

    // TODO: custom kernel to be replaced by cub::DeviceReduce::Sum when cub version is bumped to
    // v2.1.0
    detail::size<cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
        storage_.ref(), is_filled, counter.data());

    return counter.load_to_host(stream);
  }

  /**
   * @brief Gets the maximum number of elements the container can hold.
   *
   * @return The maximum number of elements the container can hold
   */
  [[nodiscard]] constexpr auto capacity() const noexcept { return storage_.capacity(); }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] constexpr key_type empty_key_sentinel() const noexcept
  {
    return empty_key_sentinel_;
  }

  /**
   * @brief Gets the key comparator.
   *
   * @return The comparator used to compare keys
   */
  [[nodiscard]] constexpr key_equal key_eq() const noexcept { return predicate_; }

  /**
   * @brief Gets the probing scheme.
   *
   * @return The probing scheme used for the container
   */
  [[nodiscard]] constexpr probing_scheme_type const& probing_scheme() const noexcept
  {
    return probing_scheme_;
  }

  /**
   * @brief Gets the container allocator.
   *
   * @return The container allocator
   */
  [[nodiscard]] constexpr allocator_type allocator() const noexcept { return storage_.allocator(); }

  /**
   * @brief Gets the non-owning storage ref.
   *
   * @return The non-owning storage ref of the container
   */
  [[nodiscard]] constexpr storage_ref_type storage_ref() const noexcept { return storage_.ref(); }

 protected:
  key_type empty_key_sentinel_;         ///< Key value that represents an empty slot
  value_type empty_slot_sentinel_;      ///< Slot value that represents an empty slot
  key_equal predicate_;                 ///< Key equality binary predicate
  probing_scheme_type probing_scheme_;  ///< Probing scheme
  storage_type storage_;                ///< Slot window storage
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
