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
#include <cuco/detail/error.hpp>
#include <cuco/detail/hyperloglog/finalizer.cuh>
#include <cuco/detail/hyperloglog/hyperloglog_ref.cuh>
#include <cuco/detail/hyperloglog/kernels.cuh>
#include <cuco/detail/hyperloglog/storage.cuh>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/detail/utils.hpp>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/allocator.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cstddef>
#include <iterator>
#include <memory>

namespace cuco::detail {
/**
 * @brief A GPU-accelerated utility for approximating the number of distinct items in a multiset.
 *
 * @note This class implements the HyperLogLog/HyperLogLog++ algorithm:
 * https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
 * @note The `Precision` parameter can be used to trade runtime/memory footprint for better
 * accuracy. A higher value corresponds to a more accurate result, however, setting the precision
 * too high will result in deminishing results.
 *
 * @tparam T Type of items to count
 * @tparam Precision Tuning parameter to trade runtime/memory footprint for better accuracy
 * @tparam Scope The scope in which operations will be performed by individual threads
 * @tparam Hash Hash function used to hash items
 * @tparam Allocator Type of allocator used for device storage
 */
template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
class hyperloglog {
 public:
  static constexpr auto thread_scope = Scope;      ///< CUDA thread scope
  static constexpr auto precision    = Precision;  ///< Precision

  template <cuda::thread_scope NewScope = thread_scope>
  using ref_type = hyperloglog_ref<T, Precision, NewScope, Hash>;  ///< Non-owning reference
                                                                   ///< type

  using value_type   = typename ref_type<>::value_type;    ///< Type of items to count
  using storage_type = typename ref_type<>::storage_type;  ///< Storage type
  using hash_type    = typename ref_type<>::hash_type;     ///< Hash function type
  using allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<storage_type>;  ///< Allocator
                                                                                     ///< type

  /**
   * @brief Constructs a `hyperloglog` host object.
   *
   * @note This function synchronizes the given stream.
   *
   * @param hash The hash function used to hash items
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the object
   */
  constexpr hyperloglog(Hash const& hash, Allocator const& alloc, cuco::cuda_stream_ref stream)
    : hash_{hash},
      allocator_{alloc},
      deleter_{1ull, allocator_},
      storage_{allocator_.allocate(1ull), deleter_}
  {
    this->clear_async(stream);
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
  void clear_async(cuco::cuda_stream_ref stream) noexcept
  {
    auto constexpr block_size = 1024;
    cuco::hyperloglog_ns::detail::clear<<<1, block_size, 0, stream>>>(this->ref());
  }

  /**
   * @brief Resets the estimator, i.e., clears the current count estimate.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `clear_async`.
   *
   * @param stream CUDA stream this operation is executed in
   */
  void clear(cuco::cuda_stream_ref stream)
  {
    this->clear_async(stream);
    stream.synchronize();
  }

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
  void add_async(InputIt first, InputIt last, cuco::cuda_stream_ref stream)
  {
    auto const num_items = cuco::detail::distance(first, last);
    if (num_items == 0) { return; }

    int grid_size         = 0;
    int block_size        = 0;
    int const shmem_bytes = sizeof(storage_type);

    // We make use of the occupancy calculator here to get the minimum number of blocks which still
    // saturate the GPU. This reduces the atomic contention on the final register array during the
    // merge phase.
    CUCO_CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
      &grid_size,
      &block_size,
      &cuco::hyperloglog_ns::detail::add_shmem<InputIt, ref_type<>>,
      shmem_bytes));

    if (grid_size != 0) {  // use shmem codepath
      cuco::hyperloglog_ns::detail::add_shmem<<<grid_size, block_size, shmem_bytes, stream>>>(
        first, num_items, this->ref());
    } else {  // use gmem codepath since there is not enough shmem available
      block_size = 0;
      CUCO_CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
        &grid_size, &block_size, &cuco::hyperloglog_ns::detail::add_gmem<InputIt, ref_type<>>));
      CUCO_EXPECTS(grid_size != 0, "Invalid kernel launch configuration");

      cuco::hyperloglog_ns::detail::add_gmem<<<grid_size, block_size, 0, stream>>>(
        first, num_items, this->ref());
    }
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
  void add(InputIt first, InputIt last, cuco::cuda_stream_ref stream)
  {
    this->add_async(first, last, stream);
    stream.synchronize();
  }

  /**
   * @brief Asynchronously merges the result of `other` estimator into `*this` estimator.
   *
   * @tparam OtherScope Thread scope of `other` estimator
   * @tparam OtherAllocator Allocator type of `other` estimator
   *
   * @param other Other estimator to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope, class OtherAllocator>
  void merge_async(hyperloglog<T, Precision, OtherScope, Hash, OtherAllocator> const& other,
                   cuco::cuda_stream_ref stream) noexcept
  {
    this->merge_async(other.ref(), stream);
  }

  /**
   * @brief Merges the result of `other` estimator into `*this` estimator.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `merge_async`.
   *
   * @tparam OtherScope Thread scope of `other` estimator
   * @tparam OtherAllocator Allocator type of `other` estimator
   *
   * @param other Other estimator to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope, class OtherAllocator>
  void merge(hyperloglog<T, Precision, OtherScope, Hash, OtherAllocator> const& other,
             cuco::cuda_stream_ref stream)
  {
    this->merge_async(other, stream);
    stream.synchronize();
  }

  /**
   * @brief Asynchronously merges the result of `other` estimator reference into `*this` estimator.
   *
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param other Other estimator reference to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope>
  void merge_async(ref_type<OtherScope> const& other, cuco::cuda_stream_ref stream) noexcept
  {
    auto constexpr block_size = 1024;
    cuco::hyperloglog_ns::detail::merge<<<1, block_size, 0, stream>>>(other, this->ref());
  }

  /**
   * @brief Merges the result of `other` estimator reference into `*this` estimator.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `merge_async`.
   *
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param other Other estimator reference to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope>
  void merge(ref_type<OtherScope> const& other, cuco::cuda_stream_ref stream)
  {
    this->merge_async(other, stream);
    stream.synchronize();
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
  [[nodiscard]] std::size_t estimate(cuco::cuda_stream_ref stream) const
  {
    // TODO remove test code
    // std::size_t* result;
    // cudaMallocHost(&result, sizeof(std::size_t));

    // int grid_size  = 0;
    // int block_size = 0;
    // // TODO check cuda error?
    // cudaOccupancyMaxPotentialBlockSize(
    //   &grid_size, &block_size, &cuco::hyperloglog_ns::detail::estimate<ref_type<>>);

    // cuco::hyperloglog_ns::detail::estimate<<<grid_size, block_size, 0, stream>>>(
    //   result, this->ref());
    // stream.synchronize();

    // return *result;

    // TODO this function currently copies the registers to the host and then finalizes the result;
    // move computation to device? Edit: host computation is faster -.-
    storage_type registers;
    // TODO check if storage is host accessible
    CUCO_CUDA_TRY(cudaMemcpyAsync(
      &registers, this->storage_.get(), sizeof(storage_type), cudaMemcpyDeviceToHost, stream));
    stream.synchronize();

    using fp_type = typename ref_type<>::fp_type;
    fp_type sum   = 0;
    int zeroes    = 0;

    // geometric mean computation + count registers with 0s
    for (int i = 0; i < registers.size(); ++i) {
      auto const reg = registers[i];
      sum += fp_type{1} / static_cast<fp_type>(1ull << reg);
      zeroes += reg == 0;
    }

    // pass intermediate result to finalizer for bias correction, etc.
    return cuco::hyperloglog_ns::detail::finalizer<Precision>::finalize(sum, zeroes);
  }

  /**
   * @brief Get device ref.
   *
   * @return Device ref object of the current `distinct_count_estimator` host object
   */
  [[nodiscard]] ref_type<> ref() const noexcept
  {
    return ref_type<>{*(this->storage_.get()), this->hash_};
  }

  /**
   * @brief Get storage ref.
   *
   * @return Reference to storage
   */
  [[nodiscard]] storage_type& storage_ref() const noexcept { return *(this->storage_.get()); }

  /**
   * @brief Get hash function.
   *
   * @return The hash function
   */
  [[nodiscard]] auto hash() const noexcept { return this->hash_; }

 private:
  hash_type hash_;                                       ///< Hash function used to hash items
  allocator_type allocator_;                             ///< Storage allocator
  custom_deleter<std::size_t, allocator_type> deleter_;  ///< Storage deleter
  std::unique_ptr<storage_type, custom_deleter<std::size_t, allocator_type>> storage_;  ///< Storage

  // Needs to be friends with other instantiations of this class template to have access to their
  // storage
  template <class T_, int32_t Precision_, cuda::thread_scope Scope_, class Hash_, class Allocator_>
  friend class hyperloglog;
};
}  // namespace cuco::detail