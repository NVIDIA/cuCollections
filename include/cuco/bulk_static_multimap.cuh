/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuco/bulk_static_multimap_ref.cuh>
#include <cuco/detail/bulk_static_multimap/kernels.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/allocator.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda/std/functional>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace cuco {
/**
 * @brief Duplicate-preserving GPU multimap constructed from a complete input range.
 *
 * No incremental insertion or erase is supported. Every build replaces all entries and
 * invalidates old refs. Rebuild and destruction require external ordering after every reader.
 * Construction uses a device-wide histogram, scan and scatter; device collective construction
 * into caller-owned storage is available separately in bulk_static_multimap_builder.cuh.
 *
 * No key or mapped value is reserved in storage. The caller-supplied empty_value is only the
 * output marker for host find misses. Keys and values must form a trivially copyable pair.
 * Hash and equality must agree; equivalent keys hash identically. Output ordering is unspecified.
 * Allocations and destruction use the constructor stream, which must outlive this object.
 * Work on other streams must be explicitly ordered after construction and before destruction.
 */
template <class Key,
          class T,
          class KeyEqual  = cuda::std::equal_to<Key>,
          class Hash      = cuco::default_hash_function<Key>,
          class Allocator = cuco::cuda_allocator<cuco::pair<Key, T>>>
class bulk_static_multimap {
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
   * @brief Preallocates record capacity and bucket metadata; initializes an empty table.
   * A zero bucket count chooses ceil(capacity/8), with at least one bucket.
   * Capacity is exact and counts occurrences; bucket_count controls query scan length.
   * The constructor enqueues initialization without synchronizing the supplied stream.
   */
  explicit bulk_static_multimap(size_type capacity,
                                T empty_value,
                                size_type bucket_count  = 0,
                                KeyEqual equal          = {},
                                Hash hash               = {},
                                Allocator allocator     = {},
                                cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}})
    : capacity_{checked_capacity(capacity)},
      buckets_{checked_buckets(capacity, bucket_count)},
      empty_value_{empty_value},
      equal_{equal},
      hash_{hash},
      allocator_{allocator},
      records_{capacity_, allocator, stream},
      offsets_{buckets_ + 1, allocator, stream},
      counters_{buckets_ + 1, allocator, stream},
      scan_bytes_{scan_workspace_size(counters_.data(), offsets_.data(), buckets_ + 1, stream)},
      scan_workspace_{scan_bytes_, allocator, stream}
  {
    CUCO_CUDA_TRY(
      cudaMemsetAsync(offsets_.data(), 0, (buckets_ + 1) * sizeof(size_type), stream.get()));
  }

  bulk_static_multimap(bulk_static_multimap const&)            = delete;
  bulk_static_multimap& operator=(bulk_static_multimap const&) = delete;
  bulk_static_multimap(bulk_static_multimap&&)                 = default;
  bulk_static_multimap& operator=(bulk_static_multimap&&)      = delete;

  size_type size() const noexcept { return size_; }
  size_type capacity() const noexcept { return capacity_; }
  size_type bucket_count() const noexcept { return buckets_; }

  template <class... Operators>
  auto ref(Operators...) const noexcept
  {
    return bulk_static_multimap_ref<Key, T, size_type, KeyEqual, Hash, Operators...>{
      records_.data(), offsets_.data(), size_, buckets_, equal_, hash_};
  }

  /** @brief Replaces contents from [first,last); inputs must be disjoint from retained storage. */
  template <class InputIt>
  void build_async(InputIt first,
                   InputIt last,
                   cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}})
  {
    auto const count = detail::bulk_multimap_range_size(first, last);
    CUCO_EXPECTS(count <= capacity_, "Input exceeds multimap record capacity");
    auto* counters     = counters_.data();
    auto* offsets      = offsets_.data();
    auto* records      = records_.data();
    auto const buckets = buckets_;
    auto const hash    = hash_;
    CUCO_CUDA_TRY(cudaMemsetAsync(counters, 0, (buckets + 1) * sizeof(size_type), stream.get()));
    detail::bulk_multimap_launch(count, stream, [=] __device__(size_type i) {
      value_type const pair = first[i];
      auto const b          = detail::bulk_bucket(hash, pair.first, buckets);
      cuda::atomic_ref<size_type, cuda::thread_scope_device>{counters[b]}.fetch_add(
        1, cuda::memory_order_relaxed);
    });
    auto bytes = scan_bytes_;
    CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      scan_workspace_.data(), bytes, counters, offsets, buckets + 1, stream.get()));
    CUCO_CUDA_TRY(cudaMemsetAsync(counters, 0, buckets * sizeof(size_type), stream.get()));
    detail::bulk_multimap_launch(count, stream, [=] __device__(size_type i) {
      value_type const pair = first[i];
      auto const b          = detail::bulk_bucket(hash, pair.first, buckets);
      auto const position =
        cuda::atomic_ref<size_type, cuda::thread_scope_device>{counters[b]}.fetch_add(
          1, cuda::memory_order_relaxed);
      records[offsets[b] + position] = pair;
    });
    size_ = count;
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
    auto const n = checked_probe_size(first, last);
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

  template <class InputIt, class Callback>
  void for_each_async(InputIt first,
                      InputIt last,
                      Callback callback,
                      cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    auto const view = ref(cuco::for_each);
    detail::bulk_multimap_launch(
      detail::bulk_multimap_range_size(first, last), stream, [=] __device__(size_type i) {
        view.for_each(first[i], [=] __device__(auto it) { callback(value_type{*it}); });
      });
  }

  template <class InputIt, class Callback>
  void for_each(InputIt first,
                InputIt last,
                Callback callback,
                cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    for_each_async(first, last, callback, stream);
    stream.sync();
  }

  template <class Callback>
  void for_each_async(Callback callback,
                      cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    auto const* records = records_.data();
    detail::bulk_multimap_launch(
      size_, stream, [=] __device__(size_type i) { callback(value_type{records[i]}); });
  }

  template <class Callback>
  void for_each(Callback callback,
                cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    for_each_async(callback, stream);
    stream.sync();
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
    auto const n = checked_probe_size(first, last);
    if (!n || !size_) {
      stream.sync();
      return {probes, matches};
    }
    auto const view = ref(cuco::count, cuco::for_each);
    detail::bulk_multimap_buffer<size_type, Allocator> counts{n + 1, allocator_, stream};
    detail::bulk_multimap_buffer<size_type, Allocator> positions{n + 1, allocator_, stream};
    auto* sizes  = counts.data();
    auto* starts = positions.data();
    detail::bulk_multimap_launch(
      n + 1, stream, [=] __device__(size_type i) { sizes[i] = i < n ? view.count(first[i]) : 0; });
    auto bytes = scan_workspace_size(sizes, starts, n + 1, stream);
    detail::bulk_multimap_buffer<std::byte, Allocator> scratch{bytes, allocator_, stream};
    CUCO_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(scratch.data(), bytes, sizes, starts, n + 1, stream.get()));
    detail::bulk_multimap_launch(n, stream, [=] __device__(size_type i) {
      auto position = starts[i];
      view.for_each(first[i], [&] __device__(auto it) {
        probes[position]    = first[i];
        matches[position++] = *it;
      });
    });
    size_type result;
    CUCO_CUDA_TRY(
      cudaMemcpyAsync(&result, starts + n, sizeof(result), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    if (!result) return {probes, matches};
    return {probes + result, matches + result};
  }

  /** @brief Output buffers must hold size() entries. Synchronizes the stream. */
  template <class KeyOutput, class ValueOutput>
  std::pair<KeyOutput, ValueOutput> retrieve_all(KeyOutput keys,
                                                 ValueOutput values,
                                                 cuda::stream_ref stream = cuda::stream_ref{
                                                   cudaStream_t{nullptr}}) const
  {
    auto const* records = records_.data();
    detail::bulk_multimap_launch(size_, stream, [=] __device__(size_type i) {
      keys[i]   = records[i].first;
      values[i] = records[i].second;
    });
    stream.sync();
    if (!size_) return {keys, values};
    return {keys + size_, values + size_};
  }

 private:
  static size_type checked_capacity(size_type capacity)
  {
    CUCO_EXPECTS(capacity <= static_cast<size_type>(std::numeric_limits<std::ptrdiff_t>::max()) &&
                   capacity <= std::numeric_limits<size_type>::max() / sizeof(value_type),
                 "Record capacity overflows");
    return capacity;
  }
  static size_type checked_buckets(size_type capacity, size_type buckets)
  {
    auto const result =
      buckets ? buckets : std::max(size_type{1}, capacity / 8 + (capacity % 8 != 0));
    CUCO_EXPECTS(result < std::numeric_limits<size_type>::max() / sizeof(size_type),
                 "Bucket metadata size overflows");
    return result;
  }
  static size_type scan_workspace_size(size_type* input,
                                       size_type* output,
                                       size_type size,
                                       cuda::stream_ref stream)
  {
    size_type bytes = 0;
    CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, bytes, input, output, size, stream.get()));
    return bytes;
  }
  template <class InputIt>
  size_type checked_probe_size(InputIt first, InputIt last) const
  {
    auto const n = detail::bulk_multimap_range_size(first, last);
    CUCO_EXPECTS(
      !size_ || n <= static_cast<size_type>(std::numeric_limits<std::ptrdiff_t>::max()) / size_,
      "Maximum query result size overflows");
    return n;
  }

  size_type capacity_;
  size_type buckets_;
  size_type size_ = 0;
  T empty_value_;
  KeyEqual equal_;
  Hash hash_;
  Allocator allocator_;
  detail::bulk_multimap_buffer<value_type, Allocator> records_;
  detail::bulk_multimap_buffer<size_type, Allocator> offsets_;
  detail::bulk_multimap_buffer<size_type, Allocator> counters_;
  size_type scan_bytes_;
  detail::bulk_multimap_buffer<std::byte, Allocator> scan_workspace_;
};
}  // namespace cuco
