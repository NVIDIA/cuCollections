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
#include <cuco/detail/hyperloglog/finalizer.cuh>
#include <cuco/detail/hyperloglog/hyperloglog_ref.cuh>
#include <cuco/detail/hyperloglog/kernels.cuh>
#include <cuco/detail/hyperloglog/storage.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/allocator.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cstddef>
#include <iterator>
#include <memory>

namespace cuco::detail {
template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
class hyperloglog {
 public:
  static constexpr auto thread_scope = Scope;  ///< CUDA thread scope
  static constexpr auto precision    = Precision;

  using allocator_type = Allocator;  ///< Allocator type
  using storage_type   = detail::hyperloglog_storage<precision>;
  using storage_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<storage_type>;

  template <cuda::thread_scope NewScope = thread_scope>
  using ref_type = hyperloglog_ref<T, Precision, NewScope, Hash>;

  constexpr hyperloglog(cuco::cuda_thread_scope<Scope>,
                        Hash const& hash,
                        Allocator const& alloc,
                        cuco::cuda_stream_ref stream)
    : hash_{hash},
      storage_allocator_{alloc},
      storage_deleter_{storage_allocator_},
      storage_{storage_allocator_.allocate(1ull), storage_deleter_}
  {
    this->clear_async(stream);  // TODO async or sync?
  }

  hyperloglog(hyperloglog const&)            = delete;
  hyperloglog& operator=(hyperloglog const&) = delete;
  hyperloglog(hyperloglog&&)                 = default;
  hyperloglog& operator=(hyperloglog&&)      = default;
  ~hyperloglog()                             = default;

  void clear_async(cuco::cuda_stream_ref stream) noexcept
  {
    auto constexpr block_size = 1024;
    cuco::hyperloglog_ns::detail::clear<<<1, block_size, 0, stream>>>(this->ref());
  }

  void clear(cuco::cuda_stream_ref stream)
  {
    this->clear_async(stream);
    stream.synchronize();
  }

  template <class InputIt>
  void add_async(InputIt first, InputIt last, cuco::cuda_stream_ref stream) noexcept
  {
    auto const num_items = cuco::detail::distance(first, last);  // TODO include
    if (num_items == 0) { return; }

    // TODO fallback to local memory registers in case they don't fit in shmem

    int grid_size  = 0;
    int block_size = 0;
    // TODO check cuda error?
    cudaOccupancyMaxPotentialBlockSize(
      &grid_size, &block_size, &cuco::hyperloglog_ns::detail::add_shmem<InputIt, ref_type<>>);

    cuco::hyperloglog_ns::detail::add_shmem<<<grid_size, block_size, 0, stream>>>(
      first, num_items, this->ref());
  }

  template <class InputIt>
  void add(InputIt first, InputIt last, cuco::cuda_stream_ref stream)
  {
    this->add_async(first, last, stream);
    stream.synchronize();
  }

  template <cuda::thread_scope OtherScope, class OtherAllocator>
  void merge_async(hyperloglog<T, Precision, OtherScope, Hash, OtherAllocator> const& other,
                   cuco::cuda_stream_ref stream = {}) noexcept
  {
    this->merge_async(other.ref(), stream);
  }

  template <cuda::thread_scope OtherScope, class OtherAllocator>
  void merge(hyperloglog<T, Precision, OtherScope, Hash, OtherAllocator> const& other,
             cuco::cuda_stream_ref stream = {})
  {
    this->merge_async(other, stream);
    stream.synchronize();
  }

  template <cuda::thread_scope OtherScope>
  void merge_async(ref_type<OtherScope> const& other, cuco::cuda_stream_ref stream = {}) noexcept
  {
    auto constexpr block_size = 1024;
    cuco::hyperloglog_ns::detail::merge<<<1, block_size, 0, stream>>>(other, this->ref());
  }

  template <cuda::thread_scope OtherScope>
  void merge(ref_type<OtherScope> const& other, cuco::cuda_stream_ref stream = {})
  {
    this->merge_async(other, stream);
    stream.synchronize();
  }

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
    for (std::size_t i = 0; i < registers.size(); ++i) {
      auto const reg = registers[i];
      sum += fp_type{1} / static_cast<fp_type>(1 << reg);
      zeroes += reg == 0;
    }

    // pass intermediate result to finalizer for bias correction, etc.
    return cuco::hyperloglog_ns::detail::finalizer<Precision>::finalize(sum, zeroes);
  }

  [[nodiscard]] ref_type<> ref() const noexcept
  {
    return ref_type<>{*(this->storage_.get()), {}, this->hash_};
  }

 private:
  struct storage_deleter {
    using pointer = typename storage_allocator_type::value_type*;

    storage_deleter(storage_allocator_type& a) : allocator{a} {}

    storage_deleter(storage_deleter const&) = default;

    void operator()(pointer ptr) { allocator.deallocate(ptr, 1); }

    storage_allocator_type& allocator;
  };

  Hash hash_;
  storage_allocator_type storage_allocator_;
  storage_deleter storage_deleter_;
  std::unique_ptr<storage_type, storage_deleter> storage_;

  template <class T_, int32_t Precision_, cuda::thread_scope Scope_, class Hash_, class Allocator_>
  friend class hyperloglog;
};
}  // namespace cuco::detail