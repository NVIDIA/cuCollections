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

#include <cuco/allocator.hpp>
#include <cuco/detail/pair.cuh>

#include <cstddef>
#include <memory>

namespace cuco {
namespace detail {
/**
 * @brief Base class of open addressing storage. This class should not be used directly.
 */
class storage_base {
 public:
  /**
   * @brief Constructor of base storage.
   *
   * @param size Number of slots to (de)allocate
   */
  storage_base(std::size_t size) : size_{size} {}

  /**
   * @brief Gets the total number of slots in the current storage.
   *
   * @return The total number of slots
   */
  __host__ __device__ std::size_t size() const noexcept { return size_; }

 protected:
  std::size_t size_;  ///< Total number of slots
};

/**
 * @brief Array of structure open addressing storage class.
 *
 * @tparam Key Arithmetic type used for key
 * @tparam Value Type of the mapped values
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename Key, typename T, typename Allocator>
class aos_storage : public storage_base {
 public:
  using value_type = cuco::pair<Key, T>;  ///< Type of key/value pairs
  using allocator_type =
    typename std::allocator_traits<Allocator>::rebind_alloc<value_type>;  ///< Type of the allocator
                                                                          ///< to (de)allocate slots

  /**
   * @brief Constructor of AoS storage.
   *
   * @param size Number of slots to (de)allocate
   * @param allocator Allocator used for (de)allocating device storage
   */
  aos_storage(std::size_t size, Allocator allocator)
    : storage_base{size},
      allocator_{allocator},
      delete_slots_{size_, allocator_},
      slots_{allocator_.allocate(size_), delete_slots_}
  {
  }

  aos_storage(aos_storage&&) = default;  ///< Move constructor
  /**
   * @brief Replaces the contents of the storage with another storage.
   *
   * @return Reference of the current storage object
   */
  aos_storage& operator=(aos_storage&&) = default;
  ~aos_storage()                        = default;  ///< Destructor

  aos_storage(aos_storage const&) = delete;
  aos_storage& operator=(aos_storage const&) = delete;

 private:
  /**
   * @brief Custom deleter for unique pointer of slots.
   */
  struct slot_deleter {
    slot_deleter(std::size_t size, allocator_type& alloc) : size_{size}, allocator_{alloc} {}

    slot_deleter(slot_deleter const&) = default;

    void operator()(value_type* ptr) { allocator_.deallocate(ptr, size_); }

    std::size_t size_;
    allocator_type& allocator_;
  };

 private:
  allocator_type allocator_;                         ///< Allocator used to allocate slots
  slot_deleter delete_slots_;                        ///< Custom slots deleter
  std::unique_ptr<value_type, slot_deleter> slots_;  ///< Pointer to AoS slots storage
};
}  // namespace detail
}  // namespace cuco
