/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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
#include <cuco/detail/error.hpp>
#include <cuco/detail/hyperloglog/finalizer.cuh>
#include <cuco/detail/hyperloglog/kernels.cuh>
#include <cuco/detail/utility/memcpy_async.hpp>
#include <cuco/detail/utility/strong_type.cuh>
#include <cuco/detail/utils.hpp>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuco/utility/traits.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/__algorithm/max.h>  // TODO #include <cuda/std/algorithm> once available
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <vector>

namespace cuco::detail {
CUCO_DEFINE_STRONG_TYPE(sketch_size_kb, double);
CUCO_DEFINE_STRONG_TYPE(standard_deviation, double);
CUCO_DEFINE_STRONG_TYPE(precision, int32_t);

/**
 * @brief A GPU-accelerated utility for approximating the number of distinct items in a multiset.
 *
 * @note This class implements the HyperLogLog/HyperLogLog++ algorithm:
 * https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
 *
 * @tparam T Type of items to count
 * @tparam Scope The scope in which operations will be performed by individual threads
 * @tparam Hash Hash function used to hash items
 */
template <class T, cuda::thread_scope Scope, class Hash>
class hyperloglog_impl {
  // We use `int` here since this is the smallest type that supports native `atomicMax` on GPUs
  using fp_type         = double;  ///< Floating point type used for reduction
  using hash_value_type = cuda::std::remove_cvref_t<decltype(cuda::std::declval<Hash>()(
    cuda::std::declval<T>()))>;  ///< Hash value type
  static_assert(cuda::std::is_unsigned_v<hash_value_type>,
                "HyperLogLog requires an unsigned hash value type");

 public:
  static constexpr auto thread_scope = Scope;  ///< CUDA thread scope

  using value_type    = T;                   ///< Type of items to count
  using hasher        = Hash;                ///< Hash function type
  using register_type = cuda::std::int32_t;  ///< HLL register type

  template <cuda::thread_scope NewScope>
  using with_scope = hyperloglog_impl<T, NewScope, Hash>;  ///< Ref type with different
                                                           ///< thread scope

  /**
   * @brief Constructs a non-owning `hyperloglog_impl` object.
   *
   * @throw If sketch size < 0.0625KB or 64B or standard deviation > 0.2765. Throws if called from
   * host; UB if called from device.
   * @throw If sketch storage has insufficient alignment. Throws if called from host; UB if called.
   * from device.
   *
   * @param sketch_span Reference to sketch storage
   * @param hash The hash function used to hash items
   */
  __host__ __device__ constexpr hyperloglog_impl(cuda::std::span<cuda::std::byte> sketch_span,
                                                 Hash const& hash)
    : hash_{hash},
      precision_{cuda::std::countr_zero(sketch_bytes(cuco::detail::sketch_size_kb(
                                          static_cast<double>(sketch_span.size() / 1024.0))) /
                                        sizeof(register_type))},
      sketch_{reinterpret_cast<register_type*>(sketch_span.data()),
              this->sketch_bytes() / sizeof(register_type)}
  {
#ifndef __CUDA_ARCH__
    auto const alignment =
      1ull << cuda::std::countr_zero(reinterpret_cast<cuda::std::uintptr_t>(sketch_span.data()));
    CUCO_EXPECTS(alignment >= sketch_alignment(), "Insufficient sketch alignment");

    CUCO_EXPECTS(this->precision_ >= 4, "Minimum required sketch size is 0.0625KB or 64B");
#endif
  }

  /**
   * @brief Resets the estimator, i.e., clears the current count estimate.
   *
   * @tparam CG CUDA Cooperative Group type
   *
   * @param group CUDA Cooperative group this operation is executed in
   */
  template <class CG>
  __device__ constexpr void clear(CG group) noexcept
  {
    for (int i = group.thread_rank(); i < this->sketch_.size(); i += group.size()) {
      new (&(this->sketch_[i])) register_type{};
    }
  }

  /**
   * @brief Resets the estimator, i.e., clears the current count estimate.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `clear_async`.
   *
   * @param stream CUDA stream this operation is executed in
   */
  __host__ constexpr void clear(cuda::stream_ref stream)
  {
    this->clear_async(stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  /**
   * @brief Asynchronously resets the estimator, i.e., clears the current count estimate.
   *
   * @param stream CUDA stream this operation is executed in
   */
  __host__ constexpr void clear_async(cuda::stream_ref stream) noexcept
  {
    auto constexpr block_size = 1024;
    cuco::hyperloglog_ns::detail::clear<<<1, block_size, 0, stream.get()>>>(*this);
  }

  /**
   * @brief Adds an item to the estimator.
   *
   * @param item The item to be counted
   */
  __device__ constexpr void add(T const& item) noexcept
  {
    constexpr auto hash_bits = cuda::std::numeric_limits<hash_value_type>::digits;
    auto const h             = this->hash_(item);
    auto const reg           = static_cast<int>(h >> (hash_bits - this->precision_));
    auto const w_padding = hash_value_type{1} << static_cast<hash_value_type>(this->precision_ - 1);
    auto const zeroes = cuda::std::countl_zero((h << this->precision_) | w_padding) + 1;  // __clz

    this->update_max(reg, zeroes);
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
  __host__ constexpr void add_async(InputIt first, InputIt last, cuda::stream_ref stream)
  {
    this->add_if_async(
      first, last, cuda::constant_iterator<bool>{true}, cuda::std::identity{}, stream);
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
  __host__ constexpr void add(InputIt first, InputIt last, cuda::stream_ref stream)
  {
    this->add_async(first, last, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  /**
   * @brief Asynchronously adds items in the range `[first, last)` if `pred` of the corresponding
   * stencil returns true.
   *
   * @note The item `*(first + i)` is added if `pred( *(stencil + i) )` returns true.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * T></tt> is `true`
   * @tparam StencilIt Device accessible random access iterator whose value_type is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   *
   * @param first Beginning of the sequence of items
   * @param last End of the sequence of items
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param stream CUDA stream this operation is executed in
   */
  template <class InputIt, class StencilIt, class Predicate>
  __host__ constexpr void add_if_async(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream)
  {
    auto const num_items = cuco::detail::distance(first, last);
    if (num_items == 0) { return; }

    int grid_size         = 0;
    int block_size        = 0;
    int const shmem_bytes = sketch_bytes();
    void const* kernel    = nullptr;

    if constexpr (cuda::std::contiguous_iterator<InputIt>) {
      auto const ptr                  = cuda::std::to_address(first);
      auto constexpr max_vector_bytes = 32;
      auto const alignment =
        1 << cuda::std::countr_zero(reinterpret_cast<cuda::std::uintptr_t>(ptr) | max_vector_bytes);
      auto const vector_size = alignment / sizeof(value_type);

      switch (vector_size) {
        case 2:
          kernel = reinterpret_cast<void const*>(
            cuco::hyperloglog_ns::detail::
              add_if_shmem_vectorized<2, StencilIt, Predicate, hyperloglog_impl>);
          break;
        case 4:
          kernel = reinterpret_cast<void const*>(
            cuco::hyperloglog_ns::detail::
              add_if_shmem_vectorized<4, StencilIt, Predicate, hyperloglog_impl>);
          break;
        case 8:
          kernel = reinterpret_cast<void const*>(
            cuco::hyperloglog_ns::detail::
              add_if_shmem_vectorized<8, StencilIt, Predicate, hyperloglog_impl>);
          break;
        case 16:
          kernel = reinterpret_cast<void const*>(
            cuco::hyperloglog_ns::detail::
              add_if_shmem_vectorized<16, StencilIt, Predicate, hyperloglog_impl>);
          break;
      };
    }

    if (kernel != nullptr and this->try_reserve_shmem(kernel, shmem_bytes)) {
      if constexpr (cuda::std::contiguous_iterator<InputIt>) {
        CUCO_CUDA_TRY(
          cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, shmem_bytes));

        auto const ptr      = cuda::std::to_address(first);
        void* kernel_args[] = {(void*)(&ptr),
                               (void*)(&num_items),
                               (void*)(&stencil),
                               (void*)(&pred),
                               reinterpret_cast<void*>(this)};
        CUCO_CUDA_TRY(
          cudaLaunchKernel(kernel, grid_size, block_size, kernel_args, shmem_bytes, stream.get()));
      }
    } else {
      kernel = reinterpret_cast<void const*>(
        cuco::hyperloglog_ns::detail::
          add_if_shmem<InputIt, StencilIt, Predicate, hyperloglog_impl>);
      void* kernel_args[] = {(void*)(&first),
                             (void*)(&num_items),
                             (void*)(&stencil),
                             (void*)(&pred),
                             reinterpret_cast<void*>(this)};
      if (this->try_reserve_shmem(kernel, shmem_bytes)) {
        CUCO_CUDA_TRY(
          cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, shmem_bytes));

        CUCO_CUDA_TRY(
          cudaLaunchKernel(kernel, grid_size, block_size, kernel_args, shmem_bytes, stream.get()));
      } else {
        kernel = reinterpret_cast<void const*>(
          cuco::hyperloglog_ns::detail::
            add_if_gmem<InputIt, StencilIt, Predicate, hyperloglog_impl>);

        CUCO_CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, 0));

        CUCO_CUDA_TRY(
          cudaLaunchKernel(kernel, grid_size, block_size, kernel_args, 0, stream.get()));
      }
    }
  }

  /**
   * @brief Merges the result of `other` estimator reference into `*this` estimator reference.
   *
   * @throw If this->sketch_bytes() != other.sketch_bytes() then behavior is undefined
   *
   * @tparam CG CUDA Cooperative Group type
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param group CUDA Cooperative group this operation is executed in
   * @param other Other estimator reference to be merged into `*this`
   */
  template <class CG, cuda::thread_scope OtherScope>
  __device__ constexpr void merge(CG group, hyperloglog_impl<T, OtherScope, Hash> const& other)
  {
    // TODO find a better way to do error handling in device code
    // if (other.precision_ != this->precision_) { __trap(); }

    for (int i = group.thread_rank(); i < this->sketch_.size(); i += group.size()) {
      this->update_max(i, other.sketch_[i]);
    }
  }

  /**
   * @brief Asynchronously merges the result of `other` estimator reference into `*this`
   * estimator.
   *
   * @throw If this->sketch_bytes() != other.sketch_bytes()
   *
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param other Other estimator reference to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope>
  __host__ constexpr void merge_async(hyperloglog_impl<T, OtherScope, Hash> const& other,
                                      cuda::stream_ref stream)
  {
    CUCO_EXPECTS(other.precision_ == this->precision_,
                 "Cannot merge estimators with different sketch sizes");
    auto constexpr block_size = 1024;
    cuco::hyperloglog_ns::detail::merge<<<1, block_size, 0, stream.get()>>>(other, *this);
  }

  /**
   * @brief Merges the result of `other` estimator reference into `*this` estimator.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `merge_async`.
   *
   * @throw If this->sketch_bytes() != other.sketch_bytes()
   *
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param other Other estimator reference to be merged into `*this`
   * @param stream CUDA stream this operation is executed in
   */
  template <cuda::thread_scope OtherScope>
  __host__ constexpr void merge(hyperloglog_impl<T, OtherScope, Hash> const& other,
                                cuda::stream_ref stream)
  {
    this->merge_async(other, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif
  }

  /**
   * @brief Compute the estimated distinct items count.
   *
   * @param group CUDA thread block group this operation is executed in
   *
   * @return Approximate distinct items count
   */
  [[nodiscard]] __device__ size_t
  estimate(cooperative_groups::thread_block const& group) const noexcept
  {
    __shared__ cuda::atomic<fp_type, cuda::thread_scope_block> block_sum;
    __shared__ cuda::atomic<int, cuda::thread_scope_block> block_zeroes;
    __shared__ size_t estimate;

    if (group.thread_rank() == 0) {
      new (&block_sum) decltype(block_sum){0};
      new (&block_zeroes) decltype(block_zeroes){0};
    }
    group.sync();

    fp_type thread_sum = 0;
    int thread_zeroes  = 0;
    for (int i = group.thread_rank(); i < this->sketch_.size(); i += group.size()) {
      auto const reg = this->sketch_[i];
      thread_sum += fp_type{1} / static_cast<fp_type>(1ull << reg);
      thread_zeroes += reg == 0;
    }

    // warp reduce Z and V
    auto const warp =
      cooperative_groups::tiled_partition<32, cooperative_groups::thread_block>(group);
#if defined(CUCO_HAS_CG_REDUCE_UPDATE_ASYNC)
    cooperative_groups::reduce_update_async(
      warp, block_sum, thread_sum, cooperative_groups::plus<fp_type>());
    cooperative_groups::reduce_update_async(
      warp, block_zeroes, thread_zeroes, cooperative_groups::plus<int>());
#else
    auto const warp_sum =
      cooperative_groups::reduce(warp, thread_sum, cooperative_groups::plus<fp_type>());
    auto const warp_zeroes =
      cooperative_groups::reduce(warp, thread_zeroes, cooperative_groups::plus<int>());
    // TODO warp sync needed?
    // TODO use invoke_one
    if (warp.thread_rank() == 0) {
      block_sum.fetch_add(warp_sum, cuda::std::memory_order_relaxed);
      block_zeroes.fetch_add(warp_zeroes, cuda::std::memory_order_relaxed);
    }
#endif
    group.sync();

    if (group.thread_rank() == 0) {
      auto const z        = block_sum.load(cuda::std::memory_order_relaxed);
      auto const v        = block_zeroes.load(cuda::std::memory_order_relaxed);
      auto const finalize = cuco::hyperloglog_ns::detail::finalizer(this->precision_);
      estimate            = finalize(z, v);
    }
    group.sync();

    return estimate;
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
  [[nodiscard]] __host__ size_t estimate(cuda::stream_ref stream) const
  {
    auto const num_regs = 1ull << this->precision_;
    std::vector<register_type> host_sketch(num_regs);

    // TODO check if storage is host accessible
    CUCO_CUDA_TRY(cuco::detail::memcpy_async(host_sketch.data(),
                                             this->sketch_.data(),
                                             sizeof(register_type) * num_regs,
                                             cudaMemcpyDefault,
                                             stream));
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream.sync();
#else
    stream.wait();
#endif

    fp_type sum = 0;
    int zeroes  = 0;

    // geometric mean computation + count registers with 0s
    for (auto const reg : host_sketch) {
      sum += fp_type{1} / static_cast<fp_type>(1ull << reg);
      zeroes += reg == 0;
    }

    auto const finalize = cuco::hyperloglog_ns::detail::finalizer(this->precision_);

    // pass intermediate result to finalizer for bias correction, etc.
    return finalize(sum, zeroes);
  }

  /**
   * @brief Gets the hash function.
   *
   * @return The hash function
   */
  [[nodiscard]] __host__ __device__ constexpr auto hash_function() const noexcept
  {
    return this->hash_;
  }

  /**
   * @brief Gets the span of the sketch.
   *
   * @return The cuda::std::span of the sketch
   */
  [[nodiscard]] __host__ __device__ constexpr cuda::std::span<cuda::std::byte> sketch()
    const noexcept
  {
    return cuda::std::span<cuda::std::byte>(
      reinterpret_cast<cuda::std::byte*>(this->sketch_.data()), this->sketch_bytes());
  }

  /**
   * @brief Gets the number of bytes required for the sketch storage.
   *
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] __host__ __device__ constexpr size_t sketch_bytes() const noexcept
  {
    return (1ull << this->precision_) * sizeof(register_type);
  }

  /**
   * @brief Gets the number of bytes required for the sketch storage.
   *
   * @param sketch_size_kb Upper bound sketch size in KB
   *
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] __host__ __device__ static constexpr cuda::std::size_t sketch_bytes(
    cuco::detail::sketch_size_kb sketch_size_kb) noexcept
  {
    // minimum precision is 4 or 64 bytes
    return cuda::std::max(
      static_cast<cuda::std::size_t>(sizeof(register_type) * 1ull << 4),
      cuda::std::bit_floor(static_cast<cuda::std::size_t>(sketch_size_kb * 1024)));
  }

  /**
   * @brief Gets the number of bytes required for the sketch storage.
   *
   * @param standard_deviation Upper bound standard deviation for approximation error
   *
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] __host__ __device__ static constexpr cuda::std::size_t sketch_bytes(
    cuco::detail::standard_deviation standard_deviation) noexcept
  {
    // implementation taken from
    // https://github.com/apache/spark/blob/6a27789ad7d59cd133653a49be0bb49729542abe/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/util/HyperLogLogPlusPlusHelper.scala#L43

    //  minimum precision is 4 or 64 bytes
    auto const precision = cuda::std::max(
      static_cast<cuda::std::int32_t>(4),
      static_cast<cuda::std::int32_t>(
        cuda::std::ceil(2.0 * cuda::std::log(1.106 / standard_deviation) / cuda::std::log(2.0))));

    // inverse of this function (ommitting the minimum precision constraint) is
    // standard_deviation = 1.106 / exp((precision * log(2.0)) / 2.0)

    return sizeof(register_type) * (1ull << precision);
  }

  /**
   * @brief Gets the number of bytes required for the sketch storage.
   *
   * @param precision HyperLogLog precision parameter
   *
   * @return The number of bytes required for the sketch
   */
  [[nodiscard]] __host__ __device__ static constexpr cuda::std::size_t sketch_bytes(
    cuco::detail::precision precision) noexcept
  {
    // minimum precision is 4 or 64 bytes
    auto const clamped_precision =
      cuda::std::max(cuda::std::int32_t{4}, cuda::std::int32_t{precision});
    return cuda::std::size_t{sizeof(register_type) * (1ull << clamped_precision)};
  }

  /**
   * @brief Gets the alignment required for the sketch storage.
   *
   * @return The required alignment
   */
  [[nodiscard]] __host__ __device__ static constexpr cuda::std::size_t sketch_alignment() noexcept
  {
    return cuda::std::size_t{alignof(register_type)};
  }

 private:
  /**
   * @brief Atomically updates the register at position `i` with `max(reg[i], value)`.
   *
   * @tparam Scope CUDA thread scope
   *
   * @param i Register index
   * @param value New value
   */
  __device__ constexpr void update_max(int i, register_type value) noexcept
  {
    cuda::atomic_ref<register_type, Scope> register_ref(this->sketch_[i]);
    register_ref.fetch_max(value, cuda::memory_order_relaxed);
  }

  /**
   * @brief Try expanding the shmem partition for a given kernel beyond 48KB if necessary.
   *
   * @tparam Kernel Type of kernel function
   *
   * @param kernel The kernel function
   * @param shmem_bytes Number of requested dynamic shared memory bytes
   *
   * @returns True iff kernel configuration is succesful
   */
  template <typename Kernel>
  [[nodiscard]] __host__ constexpr bool try_reserve_shmem(Kernel kernel, int shmem_bytes) const
  {
    int device = -1;
    CUCO_CUDA_TRY(cudaGetDevice(&device));
    int max_shmem_bytes = 0;
    CUCO_CUDA_TRY(
      cudaDeviceGetAttribute(&max_shmem_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    if (shmem_bytes <= max_shmem_bytes) {
      CUCO_CUDA_TRY(cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel),
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem_bytes));
      return true;
    } else {
      return false;
    }
  }

  hasher hash_;                            ///< Hash function used to hash items
  int32_t precision_;                      ///< HLL precision parameter
  cuda::std::span<register_type> sketch_;  ///< HLL sketch storage

  template <class T_, cuda::thread_scope Scope_, class Hash_>
  friend class hyperloglog_impl;
};
}  // namespace cuco::detail
