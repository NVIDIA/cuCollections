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

#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/detail/probing_scheme_base.cuh>
#include <cuco/extent.cuh>
#include <cuco/pair.cuh>
#include <cuco/probing_scheme.cuh>

#include <thrust/distance.h>
#include <thrust/tuple.h>

#include <cuda/atomic>

#include <cooperative_groups.h>

#include <cstdint>
#include <type_traits>

namespace cuco {
namespace experimental {
namespace detail {

/// Three-way insert result enum
enum class insert_result : int32_t { CONTINUE = 0, SUCCESS = 1, DUPLICATE = 2 };

/**
 * @brief Helper struct to store intermediate window probing results.
 */
struct window_probing_results {
  detail::equal_result state_;  ///< Equal result
  int32_t intra_window_index_;  ///< Intra-window index

  /**
   * @brief Constructs window_probing_results.
   *
   * @param state The three way equality result
   * @param index Intra-window index
   */
  __device__ explicit constexpr window_probing_results(detail::equal_result state,
                                                       int32_t index) noexcept
    : state_{state}, intra_window_index_{index}
  {
  }
};

/**
 * @brief Common device non-owning "ref" implementation class.
 *
 * @note This class should NOT be used directly.
 *
 * @throw If the size of the given key type is larger than 8 bytes
 * @throw If the given key type doesn't have unique object representations, i.e.,
 * `cuco::bitwise_comparable_v<Key> == false`
 * @throw If the probing scheme type is not inherited from `cuco::detail::probing_scheme_base`
 *
 * @tparam Key Type used for keys. Requires `cuco::is_bitwise_comparable_v<Key>` returning true
 * @tparam Scope The scope in which operations will be performed by individual threads.
 * @tparam KeyEqual Binary callable type used to compare two keys for equality
 * @tparam ProbingScheme Probing scheme (see `include/cuco/probing_scheme.cuh` for options)
 * @tparam StorageRef Storage ref type
 */
template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
class open_addressing_ref_impl {
  static_assert(sizeof(Key) <= 8, "Container does not support key types larger than 8 bytes.");

  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

  static_assert(
    std::is_base_of_v<cuco::experimental::detail::probing_scheme_base<ProbingScheme::cg_size>,
                      ProbingScheme>,
    "ProbingScheme must inherit from cuco::detail::probing_scheme_base");

  // TODO: how to re-enable this check?
  // static_assert(is_window_extent_v<typename StorageRef::extent_type>,
  // "Extent is not a valid cuco::window_extent");

 public:
  using key_type            = Key;                                     ///< Key type
  using probing_scheme_type = ProbingScheme;                           ///< Type of probing scheme
  using storage_ref_type    = StorageRef;                              ///< Type of storage ref
  using window_type         = typename storage_ref_type::window_type;  ///< Window type
  using value_type          = typename storage_ref_type::value_type;   ///< Storage element type
  using extent_type         = typename storage_ref_type::extent_type;  ///< Extent type
  using size_type           = typename storage_ref_type::size_type;    ///< Probing scheme size type
  using key_equal           = KeyEqual;  ///< Type of key equality binary callable
  using iterator            = typename storage_ref_type::iterator;   ///< Slot iterator type
  using const_iterator = typename storage_ref_type::const_iterator;  ///< Const slot iterator type

  static constexpr auto cg_size = probing_scheme_type::cg_size;  ///< Cooperative group size
  static constexpr auto window_size =
    storage_ref_type::window_size;  ///< Number of elements handled per window
  static constexpr auto has_payload =
    not std::is_same_v<key_type, value_type>;  ///< Determines if the container is a key/value or
                                               ///< key-only store

  /**
   * @brief Constructs open_addressing_ref_impl.
   *
   * @param empty_slot_sentinel Sentinel indicating an empty slot
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ explicit constexpr open_addressing_ref_impl(
    value_type empty_slot_sentinel,
    key_equal const& predicate,
    probing_scheme_type const& probing_scheme,
    storage_ref_type storage_ref) noexcept
    : empty_slot_sentinel_{empty_slot_sentinel},
      predicate_{
        this->extract_key(empty_slot_sentinel), this->extract_key(empty_slot_sentinel), predicate},
      probing_scheme_{probing_scheme},
      storage_ref_{storage_ref}
  {
  }

  /**
   * @brief Constructs open_addressing_ref_impl.
   *
   * @param empty_slot_sentinel Sentinel indicating an empty slot
   * @param erased_key_sentinel Sentinel indicating an erased key
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ explicit constexpr open_addressing_ref_impl(
    value_type empty_slot_sentinel,
    key_type erased_key_sentinel,
    key_equal const& predicate,
    probing_scheme_type const& probing_scheme,
    storage_ref_type storage_ref) noexcept
    : empty_slot_sentinel_{empty_slot_sentinel},
      predicate_{this->extract_key(empty_slot_sentinel), erased_key_sentinel, predicate},
      probing_scheme_{probing_scheme},
      storage_ref_{storage_ref}
  {
  }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] __host__ __device__ constexpr key_type const& empty_key_sentinel() const noexcept
  {
    return this->predicate_.empty_sentinel_;
  }

  /**
   * @brief Gets the sentinel value used to represent an empty payload slot.
   *
   * @return The sentinel value used to represent an empty payload slot
   */
  template <bool Dummy = true, typename Enable = std::enable_if_t<has_payload and Dummy>>
  [[nodiscard]] __host__ __device__ constexpr auto const& empty_value_sentinel() const noexcept
  {
    return this->extract_payload(this->empty_slot_sentinel());
  }

  /**
   * @brief Gets the sentinel value used to represent an erased key slot.
   *
   * @return The sentinel value used to represent an erased key slot
   */
  [[nodiscard]] __host__ __device__ constexpr key_type const& erased_key_sentinel() const noexcept
  {
    return this->predicate_.erased_sentinel_;
  }

  /**
   * @brief Gets the sentinel used to represent an empty slot.
   *
   * @return The sentinel value used to represent an empty slot
   */
  [[nodiscard]] __host__ __device__ constexpr value_type const& empty_slot_sentinel() const noexcept
  {
    return empty_slot_sentinel_;
  }

  /**
   * @brief Returns the function that compares keys for equality.
   *
   * @return The key equality predicate
   */
  [[nodiscard]] __device__ constexpr detail::equal_wrapper<key_type, key_equal> const& predicate()
    const noexcept
  {
    return this->predicate_;
  }

  /**
   * @brief Gets the probing scheme.
   *
   * @return The probing scheme used for the container
   */
  [[nodiscard]] __device__ constexpr probing_scheme_type const& probing_scheme() const noexcept
  {
    return probing_scheme_;
  }

  /**
   * @brief Gets the non-owning storage ref.
   *
   * @return The non-owning storage ref of the container
   */
  [[nodiscard]] __device__ constexpr storage_ref_type const& storage_ref() const noexcept
  {
    return storage_ref_;
  }

  /**
   * @brief Gets the maximum number of elements the container can hold.
   *
   * @return The maximum number of elements the container can hold
   */
  [[nodiscard]] __host__ __device__ constexpr auto capacity() const noexcept
  {
    return storage_ref_.capacity();
  }

  /**
   * @brief Returns a const_iterator to one past the last slot.
   *
   * @return A const_iterator to one past the last slot
   */
  [[nodiscard]] __host__ __device__ constexpr const_iterator end() const noexcept
  {
    return storage_ref_.end();
  }

  /**
   * @brief Returns an iterator to one past the last slot.
   *
   * @return An iterator to one past the last slot
   */
  [[nodiscard]] __host__ __device__ constexpr iterator end() noexcept { return storage_ref_.end(); }

  /**
   * @brief Inserts an element.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param value The element to insert
   *
   * @return True if the given element is successfully inserted
   */
  template <typename Value>
  __device__ bool insert(Value const& value) noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");

    auto const val    = this->heterogeneous_value(value);
    auto const key    = this->extract_key(val);
    auto probing_iter = probing_scheme_(key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      for (auto& slot_content : window_slots) {
        auto const eq_res = this->predicate_(this->extract_key(slot_content), key);

        // If the key is already in the container, return false
        if (eq_res == detail::equal_result::EQUAL) { return false; }
        if (eq_res == detail::equal_result::EMPTY or
            cuco::detail::bitwise_compare(this->extract_key(slot_content),
                                          this->erased_key_sentinel())) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          switch (attempt_insert((storage_ref_.data() + *probing_iter)->data() + intra_window_index,
                                 slot_content,
                                 val)) {
            case insert_result::CONTINUE: continue;
            case insert_result::SUCCESS: return true;
            case insert_result::DUPLICATE: return false;
          }
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Inserts an element.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   *
   * @return True if the given element is successfully inserted
   */
  template <typename Value>
  __device__ bool insert(cooperative_groups::thread_block_tile<cg_size> const& group,
                         Value const& value) noexcept
  {
    auto const val    = this->heterogeneous_value(value);
    auto const key    = this->extract_key(val);
    auto probing_iter = probing_scheme_(group, key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (this->predicate_(this->extract_key(window_slots[i]), key)) {
            case detail::equal_result::EMPTY:
              return window_probing_results{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL:
              return window_probing_results{detail::equal_result::EQUAL, i};
            default: {
              if (cuco::detail::bitwise_compare(this->extract_key(window_slots[i]),
                                                this->erased_key_sentinel())) {
                return window_probing_results{detail::equal_result::ERASED, i};
              } else {
                continue;
              }
            }
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return window_probing_results{detail::equal_result::UNEQUAL, -1};
      }();

      // If the key is already in the container, return false
      if (group.any(state == detail::equal_result::EQUAL)) { return false; }

      auto const group_contains_available =
        group.ballot(state == detail::equal_result::EMPTY or state == detail::equal_result::ERASED);
      if (group_contains_available) {
        auto const src_lane = __ffs(group_contains_available) - 1;
        auto const status =
          (group.thread_rank() == src_lane)
            ? attempt_insert((storage_ref_.data() + *probing_iter)->data() + intra_window_index,
                             window_slots[intra_window_index],
                             val)
            : insert_result::CONTINUE;

        switch (group.shfl(status, src_lane)) {
          case insert_result::SUCCESS: return true;
          case insert_result::DUPLICATE: return false;
          default: continue;
        }
      } else {
        ++probing_iter;
      }
    }
  }

  /**
   * @brief Inserts the given element into the container.
   *
   * @note This API returns a pair consisting of an iterator to the inserted element (or to the
   * element that prevented the insertion) and a `bool` denoting whether the insertion took place or
   * not.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param value The element to insert
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  template <typename Value>
  __device__ thrust::pair<iterator, bool> insert_and_find(Value const& value) noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
#if __CUDA_ARCH__ < 700
    // Spinning to ensure that the write to the value part took place requires
    // independent thread scheduling introduced with the Volta architecture.
    static_assert(
      cuco::detail::is_packable<value_type>(),
      "insert_and_find is not supported for pair types larger than 8 bytes on pre-Volta GPUs.");
#endif

    auto const val    = this->heterogeneous_value(value);
    auto const key    = this->extract_key(val);
    auto probing_iter = probing_scheme_(key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      for (auto i = 0; i < window_size; ++i) {
        auto const eq_res = this->predicate_(this->extract_key(window_slots[i]), key);
        auto* window_ptr  = (storage_ref_.data() + *probing_iter)->data();

        // If the key is already in the container, return false
        if (eq_res == detail::equal_result::EQUAL) {
          if constexpr (has_payload) {
            // wait to ensure that the write to the value part also took place
            this->wait_for_payload((window_ptr + i)->second, this->empty_slot_sentinel_.second);
          }
          return {iterator{&window_ptr[i]}, false};
        }
        if (eq_res == detail::equal_result::EMPTY or
            cuco::detail::bitwise_compare(this->extract_key(window_slots[i]),
                                          this->erased_key_sentinel())) {
          switch (this->attempt_insert_stable(window_ptr + i, window_slots[i], val)) {
            case insert_result::SUCCESS: {
              if constexpr (has_payload) {
                // wait to ensure that the write to the value part also took place
                this->wait_for_payload((window_ptr + i)->second, this->empty_slot_sentinel_.second);
              }
              return {iterator{&window_ptr[i]}, true};
            }
            case insert_result::DUPLICATE: {
              if constexpr (has_payload) {
                // wait to ensure that the write to the value part also took place
                this->wait_for_payload((window_ptr + i)->second, this->empty_slot_sentinel_.second);
              }
              return {iterator{&window_ptr[i]}, false};
            }
            default: continue;
          }
        }
      }
      ++probing_iter;
    };
  }

  /**
   * @brief Inserts the given element into the container.
   *
   * @note This API returns a pair consisting of an iterator to the inserted element (or to the
   * element that prevented the insertion) and a `bool` denoting whether the insertion took place or
   * not.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param group The Cooperative Group used to perform group insert_and_find
   * @param value The element to insert
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  template <typename Value>
  __device__ thrust::pair<iterator, bool> insert_and_find(
    cooperative_groups::thread_block_tile<cg_size> const& group, Value const& value) noexcept
  {
#if __CUDA_ARCH__ < 700
    // Spinning to ensure that the write to the value part took place requires
    // independent thread scheduling introduced with the Volta architecture.
    static_assert(
      cuco::detail::is_packable<value_type>(),
      "insert_and_find is not supported for pair types larger than 8 bytes on pre-Volta GPUs.");
#endif

    auto const val    = this->heterogeneous_value(value);
    auto const key    = this->extract_key(val);
    auto probing_iter = probing_scheme_(group, key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (this->predicate_(this->extract_key(window_slots[i]), key)) {
            case detail::equal_result::EMPTY:
              return window_probing_results{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL:
              return window_probing_results{detail::equal_result::EQUAL, i};
            default: {
              if (cuco::detail::bitwise_compare(this->extract_key(window_slots[i]),
                                                this->erased_key_sentinel())) {
                return window_probing_results{detail::equal_result::ERASED, i};
              } else {
                continue;
              }
            }
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return window_probing_results{detail::equal_result::UNEQUAL, -1};
      }();

      auto* slot_ptr = (storage_ref_.data() + *probing_iter)->data() + intra_window_index;

      // If the key is already in the container, return false
      auto const group_finds_equal = group.ballot(state == detail::equal_result::EQUAL);
      if (group_finds_equal) {
        auto const src_lane = __ffs(group_finds_equal) - 1;
        auto const res      = group.shfl(reinterpret_cast<intptr_t>(slot_ptr), src_lane);
        if (group.thread_rank() == src_lane) {
          if constexpr (has_payload) {
            // wait to ensure that the write to the value part also took place
            this->wait_for_payload(slot_ptr->second, this->empty_slot_sentinel_.second);
          }
        }
        group.sync();
        return {iterator{reinterpret_cast<value_type*>(res)}, false};
      }

      auto const group_contains_available =
        group.ballot(state == detail::equal_result::EMPTY or state == detail::equal_result::ERASED);
      if (group_contains_available) {
        auto const src_lane = __ffs(group_contains_available) - 1;
        auto const res      = group.shfl(reinterpret_cast<intptr_t>(slot_ptr), src_lane);
        auto const status   = [&, target_idx = intra_window_index]() {
          if (group.thread_rank() != src_lane) { return insert_result::CONTINUE; }
          return this->attempt_insert_stable(slot_ptr, window_slots[target_idx], val);
        }();

        switch (group.shfl(status, src_lane)) {
          case insert_result::SUCCESS: {
            if (group.thread_rank() == src_lane) {
              if constexpr (has_payload) {
                // wait to ensure that the write to the value part also took place
                this->wait_for_payload(slot_ptr->second, this->empty_slot_sentinel_.second);
              }
            }
            group.sync();
            return {iterator{reinterpret_cast<value_type*>(res)}, true};
          }
          case insert_result::DUPLICATE: {
            if (group.thread_rank() == src_lane) {
              if constexpr (has_payload) {
                // wait to ensure that the write to the value part also took place
                this->wait_for_payload(slot_ptr->second, this->empty_slot_sentinel_.second);
              }
            }
            group.sync();
            return {iterator{reinterpret_cast<value_type*>(res)}, false};
          }
          default: continue;
        }
      } else {
        ++probing_iter;
      }
    }
  }

  /**
   * @brief Erases an element.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
   *
   * @param value The element to erase
   *
   * @return True if the given element is successfully erased
   */
  template <typename ProbeKey>
  __device__ bool erase(ProbeKey const& key) noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");

    auto probing_iter = probing_scheme_(key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      for (auto& slot_content : window_slots) {
        auto const eq_res = this->predicate_(this->extract_key(slot_content), key);

        // Key doesn't exist, return false
        if (eq_res == detail::equal_result::EMPTY) { return false; }
        // Key exists, return true if successfully deleted
        if (eq_res == detail::equal_result::EQUAL) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          switch (attempt_insert((storage_ref_.data() + *probing_iter)->data() + intra_window_index,
                                 slot_content,
                                 this->erased_slot_sentinel())) {
            case insert_result::SUCCESS: return true;
            case insert_result::DUPLICATE: return false;
            default: continue;
          }
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Erases an element.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
   *
   * @param group The Cooperative Group used to perform group erase
   * @param value The element to erase
   *
   * @return True if the given element is successfully erased
   */
  template <typename ProbeKey>
  __device__ bool erase(cooperative_groups::thread_block_tile<cg_size> const& group,
                        ProbeKey const& key) noexcept
  {
    auto probing_iter = probing_scheme_(group, key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (this->predicate_(this->extract_key(window_slots[i]), key)) {
            case detail::equal_result::EMPTY:
              return window_probing_results{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL:
              return window_probing_results{detail::equal_result::EQUAL, i};
            default: continue;
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return window_probing_results{detail::equal_result::UNEQUAL, -1};
      }();

      auto const group_contains_equal = group.ballot(state == detail::equal_result::EQUAL);
      if (group_contains_equal) {
        auto const src_lane = __ffs(group_contains_equal) - 1;
        auto const status =
          (group.thread_rank() == src_lane)
            ? attempt_insert((storage_ref_.data() + *probing_iter)->data() + intra_window_index,
                             window_slots[intra_window_index],
                             this->erased_slot_sentinel())
            : insert_result::CONTINUE;

        switch (group.shfl(status, src_lane)) {
          case insert_result::SUCCESS: return true;
          case insert_result::DUPLICATE: return false;
          default: continue;
        }
      }

      // Key doesn't exist, return false
      if (group.any(state == detail::equal_result::EMPTY)) { return false; }

      ++probing_iter;
    }
  }

  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * @note If the probe key `key` was inserted into the container, returns true. Otherwise, returns
   * false.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
   *
   * @param key The key to search for
   *
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ bool contains(ProbeKey const& key) const noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
    auto probing_iter = probing_scheme_(key, storage_ref_.window_extent());

    while (true) {
      // TODO atomic_ref::load if insert operator is present
      auto const window_slots = storage_ref_[*probing_iter];

      for (auto& slot_content : window_slots) {
        switch (this->predicate_(this->extract_key(slot_content), key)) {
          case detail::equal_result::UNEQUAL: continue;
          case detail::equal_result::EMPTY: return false;
          case detail::equal_result::EQUAL: return true;
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * @note If the probe key `key` was inserted into the container, returns true. Otherwise, returns
   * false.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
   *
   * @param group The Cooperative Group used to perform group contains
   * @param key The key to search for
   *
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ bool contains(
    cooperative_groups::thread_block_tile<cg_size> const& group, ProbeKey const& key) const noexcept
  {
    auto probing_iter = probing_scheme_(group, key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      auto const state = [&]() {
        for (auto& slot : window_slots) {
          switch (this->predicate_(this->extract_key(slot), key)) {
            case detail::equal_result::EMPTY: return detail::equal_result::EMPTY;
            case detail::equal_result::EQUAL: return detail::equal_result::EQUAL;
            default: continue;
          }
        }
        return detail::equal_result::UNEQUAL;
      }();

      if (group.any(state == detail::equal_result::EQUAL)) { return true; }
      if (group.any(state == detail::equal_result::EMPTY)) { return false; }

      ++probing_iter;
    }
  }

  /**
   * @brief Finds an element in the container with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
   *
   * @param key The key to search for
   *
   * @return An iterator to the position at which the equivalent key is stored
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ const_iterator find(ProbeKey const& key) const noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
    auto probing_iter = probing_scheme_(key, storage_ref_.window_extent());

    while (true) {
      // TODO atomic_ref::load if insert operator is present
      auto const window_slots = storage_ref_[*probing_iter];

      for (auto i = 0; i < window_size; ++i) {
        switch (this->predicate_(this->extract_key(window_slots[i]), key)) {
          case detail::equal_result::EMPTY: {
            return this->end();
          }
          case detail::equal_result::EQUAL: {
            return const_iterator{&(*(storage_ref_.data() + *probing_iter))[i]};
          }
          default: continue;
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Finds an element in the container with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
   *
   * @param group The Cooperative Group used to perform this operation
   * @param key The key to search for
   *
   * @return An iterator to the position at which the equivalent key is stored
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ const_iterator find(
    cooperative_groups::thread_block_tile<cg_size> const& group, ProbeKey const& key) const noexcept
  {
    auto probing_iter = probing_scheme_(group, key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (this->predicate_(this->extract_key(window_slots[i]), key)) {
            case detail::equal_result::EMPTY:
              return window_probing_results{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL:
              return window_probing_results{detail::equal_result::EQUAL, i};
            default: continue;
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return window_probing_results{detail::equal_result::UNEQUAL, -1};
      }();

      // Find a match for the probe key, thus return an iterator to the entry
      auto const group_finds_match = group.ballot(state == detail::equal_result::EQUAL);
      if (group_finds_match) {
        auto const src_lane = __ffs(group_finds_match) - 1;
        auto const res      = group.shfl(
          reinterpret_cast<intptr_t>(&(*(storage_ref_.data() + *probing_iter))[intra_window_index]),
          src_lane);
        return const_iterator{reinterpret_cast<value_type*>(res)};
      }

      // Find an empty slot, meaning that the probe key isn't present in the container
      if (group.any(state == detail::equal_result::EMPTY)) { return this->end(); }

      ++probing_iter;
    }
  }

  /**
   * @brief Compares the content of the address `address` (old value) with the `expected` value and,
   * only if they are the same, sets the content of `address` to `desired`.
   *
   * @tparam T Address content type
   *
   * @param address The target address
   * @param expected The value expected to be found at the target address
   * @param desired The value to store at the target address if it is as expected
   *
   * @return The old value located at address `address`
   */
  template <typename T>
  __device__ constexpr auto compare_and_swap(T* address, T expected, T desired)
  {
    // temporary workaround due to performance regression
    // https://github.com/NVIDIA/libcudacxx/issues/366
    if constexpr (sizeof(T) == sizeof(unsigned int)) {
      auto* const slot_ptr           = reinterpret_cast<unsigned int*>(address);
      auto const* const expected_ptr = reinterpret_cast<unsigned int*>(&expected);
      auto const* const desired_ptr  = reinterpret_cast<unsigned int*>(&desired);
      if constexpr (Scope == cuda::thread_scope_system) {
        return atomicCAS_system(slot_ptr, *expected_ptr, *desired_ptr);
      } else if constexpr (Scope == cuda::thread_scope_device) {
        return atomicCAS(slot_ptr, *expected_ptr, *desired_ptr);
      } else if constexpr (Scope == cuda::thread_scope_block) {
        return atomicCAS_block(slot_ptr, *expected_ptr, *desired_ptr);
      } else {
        static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
      }
    } else if constexpr (sizeof(T) == sizeof(unsigned long long int)) {
      auto* const slot_ptr           = reinterpret_cast<unsigned long long int*>(address);
      auto const* const expected_ptr = reinterpret_cast<unsigned long long int*>(&expected);
      auto const* const desired_ptr  = reinterpret_cast<unsigned long long int*>(&desired);
      if constexpr (Scope == cuda::thread_scope_system) {
        return atomicCAS_system(slot_ptr, *expected_ptr, *desired_ptr);
      } else if constexpr (Scope == cuda::thread_scope_device) {
        return atomicCAS(slot_ptr, *expected_ptr, *desired_ptr);
      } else if constexpr (Scope == cuda::thread_scope_block) {
        return atomicCAS_block(slot_ptr, *expected_ptr, *desired_ptr);
      } else {
        static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
      }
    }
  }

  /**
   * @brief Atomically stores `value` at the given `address`.
   *
   * @tparam T Address content type
   *
   * @param address The target address
   * @param value The value to store
   */
  template <typename T>
  __device__ constexpr void atomic_store(T* address, T value)
  {
    if constexpr (sizeof(T) == sizeof(unsigned int)) {
      auto* const slot_ptr        = reinterpret_cast<unsigned int*>(address);
      auto const* const value_ptr = reinterpret_cast<unsigned int*>(&value);
      if constexpr (Scope == cuda::thread_scope_system) {
        atomicExch_system(slot_ptr, *value_ptr);
      } else if constexpr (Scope == cuda::thread_scope_device) {
        atomicExch(slot_ptr, *value_ptr);
      } else if constexpr (Scope == cuda::thread_scope_block) {
        atomicExch_block(slot_ptr, *value_ptr);
      } else {
        static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
      }
    } else if constexpr (sizeof(T) == sizeof(unsigned long long int)) {
      auto* const slot_ptr        = reinterpret_cast<unsigned long long int*>(address);
      auto const* const value_ptr = reinterpret_cast<unsigned long long int*>(&value);
      if constexpr (Scope == cuda::thread_scope_system) {
        atomicExch_system(slot_ptr, *value_ptr);
      } else if constexpr (Scope == cuda::thread_scope_device) {
        atomicExch(slot_ptr, *value_ptr);
      } else if constexpr (Scope == cuda::thread_scope_block) {
        atomicExch_block(slot_ptr, *value_ptr);
      } else {
        static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
      }
    }
  }

  /**
   * @brief Extracts the key from a given value type.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param value The input value
   *
   * @return The key
   */
  template <typename Value>
  [[nodiscard]] __device__ constexpr auto const& extract_key(Value const& value) const noexcept
  {
    if constexpr (this->has_payload) {
      return thrust::raw_reference_cast(value).first;
    } else {
      return thrust::raw_reference_cast(value);
    }
  }

  /**
   * @brief Extracts the payload from a given value type.
   *
   * @note This function is only available if `this->has_payload == true`
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param value The input value
   *
   * @return The payload
   */
  template <typename Value, typename Enable = std::enable_if_t<has_payload and sizeof(Value)>>
  [[nodiscard]] __device__ constexpr auto const& extract_payload(Value const& value) const noexcept
  {
    return thrust::raw_reference_cast(value).second;
  }

  /**
   * @brief Converts the given type to the container's native `value_type`.
   *
   * @tparam T Input type which is convertible to 'value_type'
   *
   * @param value The input value
   *
   * @return The converted object
   */
  template <typename T>
  [[nodiscard]] __device__ constexpr value_type native_value(T const& value) const noexcept
  {
    if constexpr (this->has_payload) {
      return {static_cast<key_type>(this->extract_key(value)), this->extract_payload(value)};
    } else {
      return static_cast<value_type>(value);
    }
  }

  /**
   * @brief Converts the given type to the container's native `value_type` while maintaining the
   * heterogeneous key type.
   *
   * @tparam T Input type which is convertible to 'value_type'
   *
   * @param value The input value
   *
   * @return The converted object
   */
  template <typename T>
  [[nodiscard]] __device__ constexpr auto heterogeneous_value(T const& value) const noexcept
  {
    if constexpr (this->has_payload and not cuda::std::is_same_v<T, value_type>) {
      using mapped_type = decltype(this->empty_slot_sentinel_.second);
      if constexpr (cuco::detail::is_cuda_std_pair_like<T>::value) {
        return cuco::pair{cuda::std::get<0>(value),
                          static_cast<mapped_type>(cuda::std::get<1>(value))};
      } else if constexpr (cuco::detail::is_thrust_pair_like<T>::value) {
        return cuco::pair{thrust::get<0>(value), static_cast<mapped_type>(thrust::get<1>(value))};
      } else {
        // hail mary (convert using .first/.second members)
        return cuco::pair{thrust::raw_reference_cast(value.first),
                          static_cast<mapped_type>(value.second)};
      }
    } else {
      return thrust::raw_reference_cast(value);
    }
  }

  /**
   * @brief Gets the sentinel used to represent an erased slot.
   *
   * @return The sentinel value used to represent an erased slot
   */
  [[nodiscard]] __device__ constexpr value_type const erased_slot_sentinel() const noexcept
  {
    if constexpr (this->has_payload) {
      return cuco::pair{this->erased_key_sentinel(), this->empty_slot_sentinel().second};
    } else {
      return this->erased_key_sentinel();
    }
  }

  /**
   * @brief Inserts the specified element with one single CAS operation.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param address Pointer to the slot in memory
   * @param expected Element to compare against
   * @param desired Element to insert
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  template <typename Value>
  [[nodiscard]] __device__ constexpr insert_result packed_cas(value_type* address,
                                                              value_type const& expected,
                                                              Value const& desired) noexcept
  {
    auto old      = compare_and_swap(address, expected, this->native_value(desired));
    auto* old_ptr = reinterpret_cast<value_type*>(&old);
    if (cuco::detail::bitwise_compare(this->extract_key(*old_ptr), this->extract_key(expected))) {
      return insert_result::SUCCESS;
    } else {
      return this->predicate_.equal_to(this->extract_key(*old_ptr), this->extract_key(desired)) ==
                 detail::equal_result::EQUAL
               ? insert_result::DUPLICATE
               : insert_result::CONTINUE;
    }
  }

  /**
   * @brief Inserts the specified element with two back-to-back CAS operations.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param address Pointer to the slot in memory
   * @param expected Element to compare against
   * @param desired Element to insert
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  template <typename Value>
  [[nodiscard]] __device__ constexpr insert_result back_to_back_cas(value_type* address,
                                                                    value_type const& expected,
                                                                    Value const& desired) noexcept
  {
    using mapped_type = decltype(this->empty_slot_sentinel_.second);

    auto const expected_key     = expected.first;
    auto const expected_payload = expected.second;

    auto old_key =
      compare_and_swap(&address->first, expected_key, static_cast<key_type>(desired.first));
    auto old_payload = compare_and_swap(&address->second, expected_payload, desired.second);

    auto* old_key_ptr     = reinterpret_cast<key_type*>(&old_key);
    auto* old_payload_ptr = reinterpret_cast<mapped_type*>(&old_payload);

    // if key success
    if (cuco::detail::bitwise_compare(*old_key_ptr, expected_key)) {
      while (not cuco::detail::bitwise_compare(*old_payload_ptr, expected_payload)) {
        old_payload = compare_and_swap(&address->second, expected_payload, desired.second);
      }
      return insert_result::SUCCESS;
    } else if (cuco::detail::bitwise_compare(*old_payload_ptr, expected_payload)) {
      atomic_store(&address->second, expected_payload);
    }

    // Our key was already present in the slot, so our key is a duplicate
    // Shouldn't use `predicate` operator directly since it includes a redundant bitwise compare
    if (this->predicate_.equal_to(*old_key_ptr, desired.first) == detail::equal_result::EQUAL) {
      return insert_result::DUPLICATE;
    }

    return insert_result::CONTINUE;
  }

  /**
   * @brief Inserts the specified element with CAS-dependent write operations.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param address Pointer to the slot in memory
   * @param expected Element to compare against
   * @param desired Element to insert
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  template <typename Value>
  [[nodiscard]] __device__ constexpr insert_result cas_dependent_write(
    value_type* address, value_type const& expected, Value const& desired) noexcept
  {
    using mapped_type = decltype(this->empty_slot_sentinel_.second);

    auto const expected_key = expected.first;

    auto old_key =
      compare_and_swap(&address->first, expected_key, static_cast<key_type>(desired.first));

    auto* old_key_ptr = reinterpret_cast<key_type*>(&old_key);

    // if key success
    if (cuco::detail::bitwise_compare(*old_key_ptr, expected_key)) {
      atomic_store(&address->second, desired.second);
      return insert_result::SUCCESS;
    }

    // Our key was already present in the slot, so our key is a duplicate
    // Shouldn't use `predicate` operator directly since it includes a redundant bitwise compare
    if (this->predicate_.equal_to(*old_key_ptr, desired.first) == detail::equal_result::EQUAL) {
      return insert_result::DUPLICATE;
    }

    return insert_result::CONTINUE;
  }

  /**
   * @brief Attempts to insert an element into a slot.
   *
   * @note Dispatches the correct implementation depending on the container
   * type and presence of other operator mixins.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param address Pointer to the slot in memory
   * @param expected Element to compare against
   * @param desired Element to insert
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  template <typename Value>
  [[nodiscard]] __device__ insert_result attempt_insert(value_type* address,
                                                        value_type const& expected,
                                                        Value const& desired) noexcept
  {
    if constexpr (sizeof(value_type) <= 8) {
      return packed_cas(address, expected, desired);
    } else {
#if (_CUDA_ARCH__ < 700)
      return cas_dependent_write(address, expected, desired);
#else
      return back_to_back_cas(address, expected, desired);
#endif
    }
  }

  /**
   * @brief Attempts to insert an element into a slot.
   *
   * @note Dispatches the correct implementation depending on the container
   * type and presence of other operator mixins.
   *
   * @note `stable` indicates that the payload will only be updated once from the sentinel value to
   * the desired value, meaning there can be no ABA situations.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param address Pointer to the slot in memory
   * @param expected Element to compare against
   * @param desired Element to insert
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  template <typename Value>
  [[nodiscard]] __device__ insert_result attempt_insert_stable(value_type* address,
                                                               value_type const& expected,
                                                               Value const& desired) noexcept
  {
    if constexpr (sizeof(value_type) <= 8) {
      return packed_cas(address, expected, desired);
    } else {
      return cas_dependent_write(address, expected, desired);
    }
  }

  /**
   * @brief Waits until the slot payload has been updated
   *
   * @note The function will return once the slot payload is no longer equal to the sentinel
   * value.
   *
   * @tparam T Map slot type
   *
   * @param slot The target slot to check payload with
   * @param sentinel The slot sentinel value
   */
  template <typename T>
  __device__ void wait_for_payload(T& slot, T const& sentinel) const noexcept
  {
    auto ref = cuda::atomic_ref<T, Scope>{slot};
    T current;
    // TODO exponential backoff strategy
    do {
      current = ref.load(cuda::std::memory_order_relaxed);
    } while (cuco::detail::bitwise_compare(current, sentinel));
  }

  // TODO: Clean up the sentinel handling since it's duplicated in ref and equal wrapper
  value_type empty_slot_sentinel_;  ///< Sentinel value indicating an empty slot
  detail::equal_wrapper<key_type, key_equal> predicate_;  ///< Key equality binary callable
  probing_scheme_type probing_scheme_;                    ///< Probing scheme
  storage_ref_type storage_ref_;                          ///< Slot storage ref
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
