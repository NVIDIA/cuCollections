/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuco/detail/bulk_static_multimap/kernels.cuh>
#include <cuco/detail/hash_csr_multimap/kernels.cuh>
#include <cuco/hash_csr_multimap_ref.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/allocator.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace cuco {
/**
 * @brief Duplicate-preserving GPU multimap constructed from a complete input range (HashCSR).
 *
 * An open-addressed table holds one 32-bit slot per distinct key. Every occurrence of a key is
 * stored contiguously in a packed record array, and a per-key offset pair delimits its range.
 * A query probes the table once, compares one stored key, and then knows the exact range of
 * matches: counting needs no scan and retrieval copies a contiguous range.
 *
 * No incremental insertion or erase is supported. Every build replaces all entries and
 * invalidates old refs. No key or mapped value is reserved in storage; `empty_value` is only the
 * output marker for host `find` misses. Output ordering is unspecified.
 */
template <class Key,
          class T,
          class KeyEqual  = cuda::std::equal_to<Key>,
          class Hash      = cuco::default_hash_function<Key>,
          class Allocator = cuco::cuda_allocator<cuco::pair<Key, T>>>
class hash_csr_multimap {
  using index_type = cuda::std::uint32_t;

 public:
  using key_type       = Key;
  using mapped_type    = T;
  using value_type     = cuco::pair<Key, T>;
  using size_type      = std::size_t;
  using key_equal      = KeyEqual;
  using hasher         = Hash;
  using allocator_type = Allocator;
  static_assert(std::is_trivially_copyable_v<value_type>);

  /**
   * @brief Preallocates storage for `capacity` records.
   *
   * `load_factor` is the largest fraction of table slots a build may occupy, reached when every
   * key is distinct. Duplicate keys share a slot.
   */
  explicit hash_csr_multimap(size_type capacity,
                             T empty_value,
                             double load_factor      = 0.5,
                             KeyEqual equal          = {},
                             Hash hash               = {},
                             Allocator allocator     = {},
                             cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}})
    : capacity_{checked_capacity(capacity)},
      slot_count_{checked_slot_count(capacity, load_factor)},
      empty_value_{empty_value},
      equal_{equal},
      hash_{hash},
      allocator_{allocator},
      records_{capacity_, allocator, stream},
      slots_{slot_count_, allocator, stream},
      offsets_{capacity_ + 1, allocator, stream},
      representatives_{capacity_, allocator, stream},
      scan_bytes_{scan_workspace_size(offsets_.data(), capacity_ + 1, stream)},
      scan_workspace_{scan_bytes_, allocator, stream}
  {
  }

  hash_csr_multimap(hash_csr_multimap const&)            = delete;
  hash_csr_multimap& operator=(hash_csr_multimap const&) = delete;
  hash_csr_multimap(hash_csr_multimap&&)                 = default;
  hash_csr_multimap& operator=(hash_csr_multimap&&)      = delete;

  size_type size() const noexcept { return size_; }
  size_type capacity() const noexcept { return capacity_; }
  size_type slot_count() const noexcept { return slot_count_; }

  template <class... Operators>
  auto ref(Operators...) const noexcept
  {
    return hash_csr_multimap_ref<Key, T, KeyEqual, Hash, Operators...>{
      table(), offsets_.data(), records_.data(), static_cast<index_type>(size_), equal_, hash_};
  }

  /** @brief Replaces contents from [first,last); inputs must be disjoint from retained storage. */
  template <class InputIt>
  void build_async(InputIt first,
                   InputIt last,
                   cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}})
  {
    auto const count = detail::bulk_multimap_range_size(first, last);
    CUCO_EXPECTS(count <= capacity_, "Input exceeds multimap record capacity");
    size_       = count;
    index_mask_ = (index_type{1} << cuda::std::bit_width(static_cast<index_type>(count))) - 1;
    if (!count) { return; }

    auto* offsets         = offsets_.data();
    auto* representatives = representatives_.data();
    CUCO_CUDA_TRY(cudaMemsetAsync(
      slots_.data(), 0xff, slot_count_ * sizeof(detail::hash_csr_slot_type), stream.get()));
    CUCO_CUDA_TRY(cudaMemsetAsync(offsets, 0, (count + 1) * sizeof(index_type), stream.get()));

    auto const size = static_cast<cuda::std::int64_t>(count);
    auto const grid = detail::hash_csr_grid_size(size);
    detail::hash_csr_build_count<<<grid, detail::hash_csr_block_size, 0, stream.get()>>>(
      size, first, table(), offsets, representatives, equal_, hash_);
    CUCO_CUDA_TRY(cudaGetLastError());

    auto bytes = scan_bytes_;
    CUCO_CUDA_TRY(cub::DeviceScan::InclusiveSum(
      scan_workspace_.data(), bytes, offsets, offsets, count + 1, stream.get()));

    detail::hash_csr_build_fill<<<grid, detail::hash_csr_block_size, 0, stream.get()>>>(
      size, first, representatives, offsets, records_.data());
    CUCO_CUDA_TRY(cudaGetLastError());
  }

  template <class InputIt>
  void build(InputIt first,
             InputIt last,
             cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}})
  {
    build_async(first, last, stream);
    stream.sync();
  }

  template <class InputIt, class OutputIt>
  void contains_async(InputIt first,
                      InputIt last,
                      OutputIt output,
                      cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    auto const view = ref(cuco::contains);
    detail::bulk_multimap_launch(
      detail::bulk_multimap_range_size(first, last), stream, [=] __device__(size_type i) {
        output[i] = view.contains(first[i]);
      });
  }

  template <class InputIt, class OutputIt>
  void contains(InputIt first,
                InputIt last,
                OutputIt output,
                cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    contains_async(first, last, output, stream);
    stream.sync();
  }

  template <class InputIt, class OutputIt>
  void find_async(InputIt first,
                  InputIt last,
                  OutputIt output,
                  cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    auto const view    = ref(cuco::find);
    auto const missing = empty_value_;
    detail::bulk_multimap_launch(
      detail::bulk_multimap_range_size(first, last), stream, [=] __device__(size_type i) {
        auto const it = view.find(first[i]);
        output[i]     = it == view.end() ? missing : it->second;
      });
  }

  template <class InputIt, class OutputIt>
  void find(InputIt first,
            InputIt last,
            OutputIt output,
            cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    find_async(first, last, output, stream);
    stream.sync();
  }

  /** @brief Sum of multiplicities across probes; repeated probes repeat their contribution. */
  template <class InputIt>
  size_type count(InputIt first,
                  InputIt last,
                  cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    auto const n = detail::bulk_multimap_range_size(first, last);
    if (!n || !size_) {
      stream.sync();
      return 0;
    }
    auto const view = ref(cuco::count);
    return thrust::transform_reduce(
      thrust::cuda::par.on(stream.get()),
      first,
      last,
      [=] __device__(auto const& key) -> size_type { return view.count(key); },
      size_type{0},
      cuda::std::plus<size_type>{});
  }

  /** @brief Output buffers must hold count(first,last) entries. Synchronizes the stream. */
  template <class InputIt, class ProbeOutput, class MatchOutput>
  std::pair<ProbeOutput, MatchOutput> retrieve(InputIt first,
                                               InputIt last,
                                               ProbeOutput probes,
                                               MatchOutput matches,
                                               cuda::stream_ref stream = cuda::stream_ref{
                                                 cudaStream_t{nullptr}}) const
  {
    auto const n = detail::bulk_multimap_range_size(first, last);
    if (!n || !size_) {
      stream.sync();
      return {probes, matches};
    }
    auto const view = ref();
    detail::bulk_multimap_buffer<index_type, Allocator> begins_buffer{n, allocator_, stream};
    detail::bulk_multimap_buffer<index_type, Allocator> sizes_buffer{n, allocator_, stream};
    detail::bulk_multimap_buffer<cuda::std::int64_t, Allocator> offsets_buffer{
      n + 1, allocator_, stream};
    auto* begins  = begins_buffer.data();
    auto* sizes   = sizes_buffer.data();
    auto* offsets = offsets_buffer.data();
    detail::bulk_multimap_launch(n, stream, [=] __device__(size_type i) {
      auto const range = view.equal_range(first[i]);
      begins[i]        = range.first;
      sizes[i]         = range.second - range.first;
    });

    auto const padded_sizes = cuda::transform_iterator(
      cuda::counting_iterator<size_type>{0},
      [=] __device__(size_type i) -> cuda::std::int64_t { return i < n ? sizes[i] : 0; });
    std::size_t bytes = 0;
    CUCO_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(nullptr, bytes, padded_sizes, offsets, n + 1, stream.get()));
    detail::bulk_multimap_buffer<std::byte, Allocator> scratch{bytes, allocator_, stream};
    CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      scratch.data(), bytes, padded_sizes, offsets, n + 1, stream.get()));

    cuda::std::int64_t result{};
    CUCO_CUDA_TRY(
      cudaMemcpyAsync(&result, offsets + n, sizeof(result), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    detail::hash_csr_launch_retrieve(result,
                                     static_cast<cuda::std::int64_t>(n),
                                     offsets,
                                     begins,
                                     first,
                                     records_.data(),
                                     probes,
                                     matches,
                                     stream);
    stream.sync();
    if (!result) { return {probes, matches}; }
    return {probes + result, matches + result};
  }

 private:
  static size_type checked_capacity(size_type capacity)
  {
    CUCO_EXPECTS(capacity <= static_cast<size_type>(std::numeric_limits<std::int32_t>::max()),
                 "Record capacity exceeds the 32-bit slot index range");
    return capacity;
  }

  static size_type checked_slot_count(size_type capacity, double load_factor)
  {
    CUCO_EXPECTS(load_factor > 0. && load_factor <= 1., "Load factor must be in (0, 1]");
    // Keep one empty slot even when every key is distinct, so that a failed probe terminates.
    auto const slots = std::max(static_cast<long double>(capacity) + 1,
                                std::ceil(static_cast<long double>(capacity) / load_factor));
    CUCO_EXPECTS(slots <= std::numeric_limits<index_type>::max(),
                 "Slot count exceeds the 32-bit slot range");
    return static_cast<size_type>(slots);
  }

  static size_type scan_workspace_size(index_type* offsets, size_type size, cuda::stream_ref stream)
  {
    size_type bytes = 0;
    CUCO_CUDA_TRY(
      cub::DeviceScan::InclusiveSum(nullptr, bytes, offsets, offsets, size, stream.get()));
    return bytes;
  }

  detail::hash_csr_table_ref table() const noexcept
  {
    return {slots_.data(), static_cast<index_type>(slot_count_), index_mask_};
  }

  size_type capacity_;
  size_type slot_count_;
  size_type size_        = 0;
  index_type index_mask_ = 0;
  T empty_value_;
  KeyEqual equal_;
  Hash hash_;
  Allocator allocator_;
  detail::bulk_multimap_buffer<value_type, Allocator> records_;
  detail::bulk_multimap_buffer<detail::hash_csr_slot_type, Allocator> slots_;
  detail::bulk_multimap_buffer<index_type, Allocator> offsets_;
  detail::bulk_multimap_buffer<index_type, Allocator> representatives_;
  size_type scan_bytes_;
  detail::bulk_multimap_buffer<std::byte, Allocator> scan_workspace_;
};
}  // namespace cuco
