/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

#include <cuco/hash_functions.cuh>
#include <cuco/static_map.cuh>
#include <cuco/types.cuh>

#include <cuda/std/functional>

#include <cstddef>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuco {

/**
 * @brief A GPU-accelerated, unordered, associative container of key-value
 * pairs with unique keys.
 *
 * This container automatically grows its capacity as necessary until device memory runs out.
 *
 * @tparam Key The type of the keys.
 * @tparam T The type of the mapped values.
 * @tparam Extent The type representing the extent of the container.
 * @tparam Scope The thread scope for the container's operations.
 * @tparam KeyEqual The equality comparison function for keys.
 * @tparam ProbingScheme The probing scheme for resolving hash collisions.
 * @tparam Allocator The allocator used for memory management.
 * @tparam Storage The storage policy for the container.
 */
template <class Key,
          class T,
          class Extent             = cuco::extent<std::size_t>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class KeyEqual           = cuda::std::equal_to<Key>,
          class ProbingScheme      = cuco::linear_probing<4,  // CG size
                                                          cuco::default_hash_function<Key>>,
          class Allocator          = cuco::cuda_allocator<cuco::pair<Key, T>>,
          class Storage            = cuco::storage<1>>
class dynamic_map {
  using map_type = static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>;

 public:
  static constexpr auto thread_scope = map_type::thread_scope;  ///< CUDA thread scope

  using key_type    = typename map_type::key_type;    ///< Key type
  using value_type  = typename map_type::value_type;  ///< Key-value pair type
  using size_type   = typename map_type::size_type;   ///< Size type
  using key_equal   = typename map_type::key_equal;   ///< Key equality comparator type
  using hasher      = typename map_type::hasher;      ///< Hash function type
  using mapped_type = T;                              ///< Payload type

  dynamic_map(dynamic_map const&)            = delete;
  dynamic_map& operator=(dynamic_map const&) = delete;

  dynamic_map(dynamic_map&&) = default;  ///< Move constructor

  /**
   * @brief Replaces the contents of the container with another container.
   *
   * @return Reference of the current map object
   */
  dynamic_map& operator=(dynamic_map&&) = default;
  ~dynamic_map()                        = default;

  /**
   * @brief Constructs a dynamically-sized map.
   *
   * The capacity of the map will automatically increase as the user adds key/value pairs using
   * `insert`.
   *
   * Capacity increases by a factor of growth_factor each time the size of the map exceeds a
   * threshold occupancy. The performance of `find` and `contains` gradually decreases each time the
   * map's capacity grows.
   *
   * @param initial_capacity The initial number of slots in the map
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param empty_value_sentinel The reserved mapped value for empty slots
   * @param pred Key equality binary predicate
   * @param probing_scheme Probing scheme
   * @param scope The scope in which operations will be performed
   * @param storage Kind of storage to use
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the map
   */
  constexpr dynamic_map(Extent initial_capacity,
                        empty_key<Key> empty_key_sentinel,
                        empty_value<T> empty_value_sentinel,
                        KeyEqual const& pred                = {},
                        ProbingScheme const& probing_scheme = {},
                        cuda_thread_scope<Scope> scope      = {},
                        Storage storage                     = {},
                        Allocator const& alloc              = {},
                        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

  /**
   * @brief Constructs a dynamically-sized map with erase capability.
   *
   * The capacity of the map will automatically increase as the user adds key/value pairs using
   * `insert`.
   *
   * Capacity increases by a factor of growth_factor each time the size of the map exceeds a
   * threshold occupancy. The performance of `find` and `contains` gradually decreases each time the
   * map's capacity grows.
   *
   * The `empty_key_sentinel` and `empty_value_sentinel` values are reserved and
   * undefined behavior results from attempting to insert any key/value pair
   * that contains either.
   *
   * @param initial_capacity The initial number of slots in the map
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param empty_value_sentinel The reserved mapped value for empty slots
   * @param erased_key_sentinel The reserved key value for erased slots
   * @param pred Key equality binary predicate
   * @param probing_scheme Probing scheme
   * @param scope The scope in which operations will be performed
   * @param storage Kind of storage to use
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the map
   *
   * @throw std::runtime error if the empty key sentinel and erased key sentinel
   * are the same value
   */
  constexpr dynamic_map(Extent initial_capacity,
                        empty_key<Key> empty_key_sentinel,
                        empty_value<T> empty_value_sentinel,
                        erased_key<Key> erased_key_sentinel,
                        KeyEqual const& pred                = {},
                        ProbingScheme const& probing_scheme = {},
                        cuda_thread_scope<Scope> scope      = {},
                        Storage storage                     = {},
                        Allocator const& alloc              = {},
                        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

  /**
   * @brief Grows the capacity of the map so there is enough space for `n` key/value pairs.
   *
   * If there is already enough space for `n` key/value pairs, the capacity remains the same.
   *
   * @param n The number of key value pairs for which there must be space
   * @param stream Stream used for executing the kernels
   */
  void reserve(size_type n, cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

  /**
   * @brief Inserts all key/value pairs in the range `[first, last)`.
   *
   * @note This function synchronizes the given stream.
   *
   * If multiple keys in `[first, last)` compare equal, it is unspecified which
   * element is inserted.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `value_type`
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt>
  void insert(InputIt first,
              InputIt last,
              cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

  /**
   * @brief For any key-value pair `{k, v}` in the range `[first, last)`, if a key equivalent to `k`
   * already exists in the map, assigns `v` to the mapped_type corresponding to the key `k`.
   * If the key does not exist, inserts the pair as if by insert.
   *
   * @note This function synchronizes the given stream.
   * @note If multiple pairs in `[first, last)` compare equal, it is unspecified which pair is
   * inserted or assigned.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * dynamic_map<K, V>::value_type></tt> is `true`
   *
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stream CUDA stream used for the operation
   */
  template <typename InputIt>
  void insert_or_assign(InputIt first,
                        InputIt last,
                        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

  /**
   * @brief Erases keys in the range `[first, last)`.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `erase_async`.
   *
   * For each key `k` in `[first, last)`, if `contains(k) == true`, removes `k` and its
   * associated value from the map. Else, no effect.
   *
   * Side-effects:
   * - `contains(k) == false`
   * - `find(k) == end()`
   * - `insert({k,v}) == true`
   * - `size()` is reduced by the total number of erased keys
   *
   * Keep in mind that `erase` does not cause the map to shrink its memory allocation.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stream Stream used for executing the kernels
   *
   * @throw std::runtime_error if a unique erased key sentinel value was not
   * provided at construction
   */
  template <typename InputIt>
  void erase(InputIt first,
             InputIt last,
             cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

  /**
   * @brief Asynchronously erases keys in the range `[first, last)`.
   *
   * For each key `k` in `[first, last)`, if `contains(k) == true`, removes `k` and its
   * associated value from the map. Else, no effect.
   *
   * @note `size()` will not be updated. Use the synchronous `erase` if you need accurate size
   * tracking.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stream Stream used for executing the kernels
   *
   * @throw std::runtime_error if a unique erased key sentinel value was not
   * provided at construction
   */
  template <typename InputIt>
  void erase_async(InputIt first,
                   InputIt last,
                   cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

  /**
   * @brief Finds the values corresponding to all keys in the range `[first, last)`.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `find_async`.
   *
   * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin +
   * i)`. Else, copies the empty value sentinel.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * convertible to the map's `mapped_type`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of values retrieved for each key
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt, typename OutputIt>
  void find(InputIt first,
            InputIt last,
            OutputIt output_begin,
            cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const;

  /**
   * @brief Asynchronously finds the values corresponding to all keys in the range `[first, last)`.
   *
   * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin +
   * i)`. Else, copies the empty value sentinel.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * convertible to the map's `mapped_type`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of values retrieved for each key
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt, typename OutputIt>
  void find_async(InputIt first,
                  InputIt last,
                  OutputIt output_begin,
                  cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const;

  /**
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `contains_async`.
   *
   * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
   *
   * @tparam InputIt Device accessible input iterator
   * @tparam OutputIt Device accessible output iterator assignable from `bool`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt, typename OutputIt>
  void contains(InputIt first,
                InputIt last,
                OutputIt output_begin,
                cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const;

  /**
   * @brief Asynchronously indicates whether the keys in the range `[first, last)` are contained in
   * the map.
   *
   * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
   *
   * @tparam InputIt Device accessible input iterator
   * @tparam OutputIt Device accessible output iterator assignable from `bool`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt, typename OutputIt>
  void contains_async(InputIt first,
                      InputIt last,
                      OutputIt output_begin,
                      cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const;

  /**
   * @brief Retrieves all of the keys and their associated values.
   *
   * @note This function synchronizes the given stream.
   *
   * The order in which keys are returned is implementation defined and not guaranteed to be
   * consistent between subsequent calls to `retrieve_all`.
   *
   * Behavior is undefined if the range beginning at `keys_out` or `values_out` is less than
   * `size()`
   *
   * @tparam KeyOut Device accessible random access output iterator whose `value_type` is
   * convertible from `key_type`.
   * @tparam ValueOut Device accessible random access output iterator whose `value_type` is
   * convertible from `mapped_type`.
   * @param keys_out Beginning output iterator for keys
   * @param values_out Beginning output iterator for values
   * @param stream CUDA stream used for this operation
   * @return Pair of iterators indicating the last elements in the output
   */
  template <typename KeyOut, typename ValueOut>
  std::pair<KeyOut, ValueOut> retrieve_all(KeyOut keys_out,
                                           ValueOut values_out,
                                           cuda::stream_ref stream = cuda::stream_ref{
                                             cudaStream_t{nullptr}}) const;

  /**
   * @brief Gets the current number of elements in the map
   *
   * @return The current number of elements in the map
   */
  [[nodiscard]] size_type size() const noexcept { return size_; }

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  [[nodiscard]] size_type capacity() const noexcept { return capacity_; }

  /**
   * @brief Gets the load factor of the hash map.
   *
   * @return The load factor of the hash map
   */
  /**
   * @brief Gets the current load factor of the map
   *
   * @return The current load factor of the map
   */
  [[nodiscard]] float load_factor() const noexcept { return static_cast<float>(size_) / capacity_; }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] constexpr key_type empty_key_sentinel() const noexcept
  {
    return submaps_.front()->empty_key_sentinel();
  }

  /**
   * @brief Gets the sentinel value used to represent an empty value slot.
   *
   * @return The sentinel value used to represent an empty value slot
   */
  [[nodiscard]] constexpr mapped_type empty_value_sentinel() const noexcept
  {
    return submaps_.front()->empty_value_sentinel();
  }

  /**
   * @brief Gets the sentinel value used to represent an erased key slot.
   *
   * @return The sentinel value used to represent an erased key slot
   */
  [[nodiscard]] constexpr key_type erased_key_sentinel() const noexcept
  {
    return submaps_.front()->erased_key_sentinel();
  }

  /**
   * @brief Gets the function used to compare keys for equality
   *
   * @return The function used to compare keys for equality
   */
  [[nodiscard]] constexpr key_equal key_eq() const noexcept { return submaps_.front()->key_eq(); }

  /**
   * @brief Gets the function(s) used to hash keys
   *
   * @return The function(s) used to hash keys
   */
  [[nodiscard]] constexpr hasher hash_function() const noexcept
  {
    return submaps_.front()->hash_function();
  }

 private:
  size_type size_;      ///< Number of keys in the map
  size_type capacity_;  ///< Capacity for next submap (also returned by capacity())

  std::vector<std::unique_ptr<map_type>> submaps_;  ///< vector of pointers to each submap
  size_type min_insert_size_;                       ///< min remaining capacity of submap for insert
  float max_load_factor_;                           ///< Maximum load factor
  Allocator alloc_;  ///< Allocator passed to submaps to allocate their device storage
};

}  // namespace cuco

#include <cuco/detail/dynamic_map/dynamic_map.inl>
