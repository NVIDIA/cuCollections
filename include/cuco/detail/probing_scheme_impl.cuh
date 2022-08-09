/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <thrust/distance.h>

#include <cooperative_groups.h>

#include <cstddef>
#include <utility>

namespace cuco {
/**
 * @brief Enum denoting whether to use window probing.
 */
enum class enable_window_probing : bool {
  YES,  ///< Enable window probing
  NO    ///< Disable window probing
};

namespace experimental {
namespace detail {

/**
 * @brief Base class of public probing scheme.
 *
 * This class should not be used directly.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam WindowSize Number of elements processed per window
 * @tparam UsesWindowProbing Whether window probing is used
 */
template <int CGSize, int WindowSize, enable_window_probing UsesWindowProbing>
class probing_scheme_base {
 public:
  /**
   * @brief The size of the CUDA cooperative thread group.
   */
  static constexpr int cg_size = CGSize;

  /**
   * @brief The number of elements processed per window.
   */
  static constexpr int window_size = WindowSize;

  /**
   * @brief The number of elements processed per window.
   */
  static constexpr enable_window_probing uses_window_probing = UsesWindowProbing;
};

/**
 * @brief Base class of probe sequence implementation.
 *
 * @tparam T Slot element type
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam WindowSize Number of elements processed per window
 * @tparam UsesWindowProbing Whether window probing is used
 */
template <int CGSize, int WindowSize, enable_window_probing UsesWindowProbing, typename SlotView>
class probing_scheme_impl_base {
 public:
  using slot_view_type = SlotView;                             ///< Slot view type
  using value_type     = typename slot_view_type::value_type;  ///< Slot element type
  /// Type of the forward iterator to `value_type`
  using iterator = value_type*;
  /// Type of the forward iterator to `const value_type`
  using const_iterator = value_type const*;

  /**
   * @brief The size of the CUDA cooperative thread group.
   */
  static constexpr int cg_size = CGSize;

  /**
   * @brief The number of elements processed per window.
   */
  static constexpr int window_size = WindowSize;

  /**
   * @brief The number of elements processed per window.
   */
  static constexpr enable_window_probing uses_window_probing = UsesWindowProbing;

  /**
   * @brief Constructs a probe sequence based on the given hash map features.
   *
   * @param slot_view View of slot storage
   */
  __host__ __device__ explicit probing_scheme_impl_base(slot_view_type slot_view)
    : slot_view_{slot_view}
  {
  }

  /**
   * @brief Returns the capacity of the hash map.
   *
   * @return The capacity of the hash map
   */
  __host__ __device__ inline std::size_t capacity() const noexcept { return slot_view_.capacity(); }

  /**
   * @brief Returns slots array.
   *
   * @return Slots array
   */
  __device__ inline iterator slots() noexcept { return slot_view_.slots(); }

  /**
   * @brief Returns slots array.
   *
   * @return Slots array
   */
  __device__ inline const_iterator slots() const noexcept { return slot_view_.slots(); }

 protected:
  slot_view_type slot_view_;  ///< Slot view
};

/**
 * @brief Cooperative Groups based double hashing scheme.
 *
 * Default probe sequence for `cuco::static_multimap`. Double hashing shows superior
 * performance when dealing with high multiplicty and/or high occupancy use cases. Performance
 * hints:
 * - `CGSize` = 1 or 2 when hash map is small (10'000'000 or less), 4 or 8 otherwise.
 *
 * `Hash1` and `Hash2` should be callable object type.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam WindowSize Window size
 * @tparam UsesWindowProbing Whether window probing is used
 * @tparam SlotView Slot storage view type
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 */
template <int CGSize,
          int WindowSize,
          enable_window_probing UsesWindowProbing,
          typename SlotView,
          typename Hash1,
          typename Hash2>
class double_hashing_impl
  : public probing_scheme_impl_base<CGSize, WindowSize, UsesWindowProbing, SlotView> {
 public:
  using probing_scheme_impl_base_type =
    probing_scheme_impl_base<CGSize, WindowSize, UsesWindowProbing, SlotView>;
  using value_type     = typename probing_scheme_impl_base_type::value_type;
  using iterator       = typename probing_scheme_impl_base_type::iterator;
  using const_iterator = typename probing_scheme_impl_base_type::const_iterator;

  using probing_scheme_impl_base_type::capacity;
  using probing_scheme_impl_base_type::cg_size;
  using probing_scheme_impl_base_type::slot_view_;
  using probing_scheme_impl_base_type::uses_window_probing;
  using probing_scheme_impl_base_type::window_size;

  /**
   * @brief Constructs a double hashing scheme based on the given hash map features.
   *
   * `hash2` takes a different seed to reduce the chance of secondary clustering.
   *
   * @param slot_view View of slot storage
   */
  __host__ __device__ explicit double_hashing_impl(SlotView slot_view)
    : probing_scheme_impl_base_type{slot_view}, hash1_{Hash1{}}, hash2_{Hash2{1}}, step_size_{0}
  {
  }

  /**
   * @brief Returns the initial slot for a given key `k`.
   *
   * If vector-load is enabled, the return slot is always a multiple of (`cg_size` * `window_size`)
   * to avoid illegal memory access.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param k The key to get the slot for
   * @return Pointer to the initial slot for `k`
   */
  template <typename ProbeKey>
  __device__ inline iterator initial_slot(ProbeKey const& k) noexcept
  {
    return const_cast<iterator>(std::as_const(*this).initial_slot(k));
  }

  /**
   * @brief Returns the initial slot for a given key `k`.
   *
   * If vector-load is enabled, the return slot is always a multiple of (`cg_size` * `window_size`)
   * to avoid illegal memory access.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param k The key to get the slot for
   * @return Pointer to the initial slot for `k`
   */
  template <typename ProbeKey>
  __device__ inline const_iterator initial_slot(ProbeKey const& k) const noexcept
  {
    auto const hash_value = hash1_(k);
    auto const capacity   = this->capacity();

    std::size_t index = 0;
    if constexpr (uses_window_probing) {
      // step size in range [1, prime - 1] * window_size
      step_size_ = (hash2_(k) % (capacity / window_size - 1) + 1) * window_size;
      index      = hash_value % (capacity / window_size) * window_size;
    } else {
      // step size in range [1, prime - 1]
      step_size_ = hash2_(k) % (capacity - 1) + 1;
      index      = hash_value % capacity;
    }
    return this->slot_view_.slots() + index;
  }

  /**
   * @brief Returns the initial slot for a given key `k`.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param g the Cooperative Group for which the initial slot is needed
   * @param k The key to get the slot for
   * @return Pointer to the initial slot for `k`
   */
  template <typename ProbeKey>
  __device__ inline iterator initial_slot(cooperative_groups::thread_block_tile<cg_size> const& g,
                                          ProbeKey const& k) noexcept
  {
    return const_cast<iterator>(std::as_const(*this).initial_slot(g, k));
  }

  /**
   * @brief Returns the initial slot for a given key `k`.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param g the Cooperative Group for which the initial slot is needed
   * @param k The key to get the slot for
   * @return Pointer to the initial slot for `k`
   */
  template <typename ProbeKey>
  __device__ inline const_iterator initial_slot(
    cooperative_groups::thread_block_tile<cg_size> const& g, ProbeKey const& k) const noexcept
  {
    if constexpr (cg_size == 1) { return initial_slot(k); }

    auto const hash_value = hash1_(k);
    auto const capacity   = this->capacity();

    std::size_t index = 0;
    if constexpr (uses_window_probing) {
      // step size in range [1, prime - 1] * cg_size * window_size
      step_size_ =
        (hash2_(k) % (capacity / (cg_size * window_size) - 1) + 1) * cg_size * window_size;
      index = hash_value % (capacity / (cg_size * window_size)) * cg_size * window_size +
              g.thread_rank() * window_size;
    } else {
      // step size in range [1, prime - 1] * cg_size
      step_size_ = (hash2_(k) % (capacity / cg_size - 1) + 1) * cg_size;
      index      = (hash_value + g.thread_rank()) % capacity;
    }
    return this->slot_view_.slots() + index;
  }

  /**
   * @brief Given a slot `s`, returns the next slot.
   *
   * If `s` is the last slot, wraps back around to the first slot.
   *
   * @param s The slot to advance
   * @return The next slot after `s`
   */
  __device__ inline iterator next_slot(iterator s) noexcept
  {
    return const_cast<iterator>(std::as_const(*this).next_slot(s));
  }

  /**
   * @brief Given a slot `s`, returns the next slot.
   *
   * If `s` is the last slot, wraps back around to the first slot.
   *
   * @param s The slot to advance
   * @return The next slot after `s`
   */
  __device__ inline const_iterator next_slot(const_iterator s) const noexcept
  {
    auto const slots  = slot_view_.slots();
    std::size_t index = thrust::distance(slots, s);
    return &slots[(index + step_size_) % capacity()];
  }

 private:
  Hash1 hash1_;                    ///< The first unary callable used to hash the key
  Hash2 hash2_;                    ///< The second unary callable used to determine step size
  mutable std::size_t step_size_;  ///< The step stride when searching for the next slot
};                                 // class double_hashing

/**
 * @brief Probing scheme used internally by hash map.
 *
 * @tparam ProbeImpl Type of probing scheme implementation
 * @tparam SlotView Type of slot storage view
 */
template <typename ProbingImpl, typename SlotView>
class probing_scheme : public ProbingImpl::template impl<SlotView> {
 public:
  using impl_type =
    typename ProbingImpl::template impl<SlotView>;  ///< Type of implementation details

  /**
   * @brief Constructs a probing scheme based on the given storage view.
   *
   * @param slot_view View of the slot storage
   */
  __host__ __device__ explicit probing_scheme(SlotView slot_view) : impl_type{slot_view} {}
};  // class probe_sequence

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
