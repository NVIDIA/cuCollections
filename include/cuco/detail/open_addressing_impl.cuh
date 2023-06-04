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
#include <cuco/detail/common_kernels.cuh>
#include <cuco/detail/storage/counter_storage.cuh>
#include <cuco/detail/tuning.cuh>
#include <cuco/extent.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/storage.cuh>
#include <cuco/utility/traits.hpp>

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
  static_assert(sizeof(Value) <= 8, "Container does not support slot types larger than 8 bytes.");

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
  using extent_type    = decltype(make_valid_extent<cg_size, window_size>(std::declval<Extent>()));
  using size_type      = typename extent_type::value_type;  ///< Size type
  using key_equal      = KeyEqual;                          ///< Key equality comparator type
  using allocator_type = Allocator;                         ///< Allocator type
  using storage_type =
    detail::storage<Storage, value_type, extent_type, allocator_type>;  ///< Storage type

  using storage_ref_type = typename storage_type::ref_type;  ///< Non-owning window storage ref type
  using probing_scheme_type = ProbingScheme;                 ///< Probe scheme type

  /**
   * @brief Constructs a statically-sized open addressing data structure with the specified initial
   * capacity, sentinel values and CUDA stream.
   *
   * The actual capacity depends on the given `capacity`, the probing scheme, CG size, and the
   * window size and it's computed via `make_valid_extent` factory. Insert operations will not
   * automatically grow the set. Attempting to insert more unique keys than the capacity of the map
   * results in undefined behavior.
   *
   * The `empty_key_sentinel` is reserved and behavior is undefined when attempting to insert
   * this sentinel value.
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
                                 KeyEqual pred                       = {},
                                 ProbingScheme const& probing_scheme = {},
                                 Allocator const& alloc              = {},
                                 cuda_stream_ref stream              = {})
    : empty_key_sentinel_{empty_key_sentinel},
      empty_slot_sentinel_{empty_slot_sentinel},
      predicate_{pred},
      probing_scheme_{probing_scheme},
      allocator_{alloc},
      storage_{make_valid_extent<cg_size, window_size>(capacity), allocator_}
  {
    storage_.initialize(this->empty_slot_sentinel_, stream);
  }

  /**
   * @brief Gets the number of elements in the container.
   *
   * @note This function synchronizes the given stream.
   *
   * @param stream CUDA stream used to get the number of inserted elements
   * @return The number of elements in the container
   */
  [[nodiscard]] size_type size(cuda_stream_ref stream = {}) const noexcept
  {
    auto counter = detail::counter_storage<size_type, thread_scope, allocator_type>{allocator_};
    counter.reset(stream);

    auto const grid_size =
      (storage_.num_windows() + detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
      (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE);

    // TODO: custom kernel to be replaced by cub::DeviceReduce::Sum when cub version is bumped to
    // v2.1.0
    detail::size<detail::CUCO_DEFAULT_BLOCK_SIZE>
      <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
        storage_.ref(), this->empty_slot_sentinel_, counter.data());

    return counter.load_to_host(stream);
  }

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
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
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] constexpr key_equal predicate() const noexcept { return predicate_; }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] constexpr probing_scheme_type probing_scheme() const noexcept
  {
    return probing_scheme_;
  }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] constexpr allocator_type allocator() const noexcept { return allocator_; }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] constexpr storage_ref_type storage_ref() const noexcept { return storage_.ref(); }

 protected:
  key_type empty_key_sentinel_;         ///< Key value that represents an empty slot
  key_equal predicate_;                 ///< Key equality binary predicate
  probing_scheme_type probing_scheme_;  ///< Probing scheme
  allocator_type allocator_;            ///< Allocator used to (de)allocate temporary storage
  storage_type storage_;                ///< Slot window storage

 private:
  value_type empty_slot_sentinel_;  ///< Value that represents an empty slot
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
