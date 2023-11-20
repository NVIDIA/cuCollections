/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cuco/detail/error.hpp>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/extent.cuh>

#include <cuda/atomic>

#include <memory>

namespace cuco {
namespace experimental {
namespace detail {
/**
 * @brief Device atomic counter storage class.
 *
 * @tparam SizeType Type of storage size
 * @tparam Scope The scope in which the counter will be used by individual threads
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename SizeType, cuda::thread_scope Scope, typename Allocator>
class counter_storage : public storage_base<cuco::experimental::extent<SizeType, 1>> {
 public:
  using storage_base<cuco::experimental::extent<SizeType, 1>>::capacity;  ///< Storage size

  using size_type      = SizeType;                        ///< Size type
  using value_type     = cuda::atomic<size_type, Scope>;  ///< Type of the counter
  using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<
    value_type>;  ///< Type of the allocator to (de)allocate counter
  using counter_deleter_type =
    custom_deleter<size_type, allocator_type>;  ///< Type of counter deleter

  /**
   * @brief Constructor of counter storage.
   *
   * @param allocator Allocator used for (de)allocating device storage
   */
  explicit constexpr counter_storage(Allocator const& allocator)
    : storage_base<cuco::experimental::extent<SizeType, 1>>{cuco::experimental::extent<size_type,
                                                                                       1>{}},
      allocator_{allocator},
      counter_deleter_{this->capacity(), allocator_},
      counter_{allocator_.allocate(this->capacity()), counter_deleter_}
  {
  }

  /**
   * @brief Asynchronously resets counter to zero.
   *
   * @param stream CUDA stream used to reset
   */
  void reset(cuda_stream_ref stream)
  {
    static_assert(sizeof(size_type) == sizeof(value_type));
    CUCO_CUDA_TRY(cudaMemsetAsync(this->data(), 0, sizeof(value_type), stream));
  }

  /**
   * @brief Gets device atomic counter pointer.
   *
   * @return Pointer to the device atomic counter
   */
  [[nodiscard]] constexpr value_type* data() noexcept { return counter_.get(); }

  /**
   * @brief Gets device atomic counter pointer.
   *
   * @return Pointer to the device atomic counter
   */
  [[nodiscard]] constexpr value_type* data() const noexcept { return counter_.get(); }

  /**
   * @brief Atomically obtains the value of the device atomic counter and copies it to the host.
   *
   * @note This API synchronizes the given `stream`.
   *
   * @param stream CUDA stream used to copy device value to the host
   * @return Value of the atomic counter
   */
  [[nodiscard]] constexpr size_type load_to_host(cuda_stream_ref stream) const
  {
    size_type h_count;
    CUCO_CUDA_TRY(
      cudaMemcpyAsync(&h_count, this->data(), sizeof(size_type), cudaMemcpyDeviceToHost, stream));
    stream.synchronize();
    return h_count;
  }

 private:
  allocator_type allocator_;              ///< Allocator used to (de)allocate counter
  counter_deleter_type counter_deleter_;  ///< Custom counter deleter
  std::unique_ptr<value_type, counter_deleter_type> counter_;  ///< Pointer to counter storage
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
