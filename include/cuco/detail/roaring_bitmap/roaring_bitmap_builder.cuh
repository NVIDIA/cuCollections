/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/detail/roaring_bitmap/roaring_bitmap_kernels.cuh>
#include <cuco/detail/roaring_bitmap/roaring_bitmap_storage.cuh>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/detail/utility/memcpy_async.hpp>
#include <cuco/detail/utils.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/stream_ref>

#include <algorithm>
#include <memory>

namespace cuco::experimental::detail {

enum class roaring_bitmap_builder_input_order { unsorted, sorted, sorted_unique };

/**
 * @brief One-shot builder for a 32-bit Roaring bitmap.
 *
 * Construction computes the temporary workspace requirement. Calling `build` allocates temporary
 * storage, executes the build, and returns the serialized bitmap storage.
 *
 * @tparam InputIt Random access iterator with `cuda::std::uint32_t` value type
 * @tparam Allocator Allocator type used for temporary and final storage
 */
template <class InputIt, class Allocator>
class roaring_bitmap_builder {
  using input_type = typename cuda::std::iterator_traits<InputIt>::value_type;
  static_assert(cuda::std::is_same_v<input_type, cuda::std::uint32_t>,
                "roaring_bitmap factories require an input iterator with uint32_t value_type");

 public:
  using storage_type = roaring_bitmap_storage<cuda::std::uint32_t, Allocator>;

  /**
   * @brief Prepares a build for the specified input range and ordering.
   *
   * @param first Beginning of the input range
   * @param last End of the input range
   * @param input_order Ordering and uniqueness guarantees for the input range
   * @param alloc Allocator used for temporary and final storage
   * @param stream Stream used for all allocation and build work
   */
  explicit roaring_bitmap_builder(InputIt first,
                                  InputIt last,
                                  roaring_bitmap_builder_input_order input_order,
                                  Allocator const& alloc,
                                  cuda::stream_ref stream)
    : first_{first},
      num_indices_{std::max<cuda::std::int64_t>(0, cuco::detail::distance(first, last))},
      num_container_slots_{static_cast<cuda::std::size_t>(
        std::min(num_indices_,
                 static_cast<cuda::std::int64_t>(
                   roaring_bitmap_metadata<cuda::std::uint32_t>::max_num_containers)))},
      input_order_{input_order},
      alloc_{alloc},
      stream_{stream},
      workspace_bytes_{compute_workspace_bytes()}
  {
  }

  /**
   * @brief Executes the prepared build.
   *
   * @return Storage containing the serialized bitmap
   */
  [[nodiscard]] storage_type build() &&
  {
    if (num_indices_ == 0) {
      return write_serialized_bitmap(first_, empty_build_state(), nullptr, nullptr, nullptr);
    }

    switch (input_order_) {
      case roaring_bitmap_builder_input_order::unsorted: return build_unsorted();
      case roaring_bitmap_builder_input_order::sorted: return build_sorted();
      case roaring_bitmap_builder_input_order::sorted_unique: return build_sorted_unique();
    }
    CUCO_FAIL("Invalid roaring_bitmap input order");
  }

 private:
  struct sorted_indices_result {
    cuda::std::uint32_t* indices;
    cuda::std::uint32_t* available_buffer;
  };

  template <class U>
  [[nodiscard]] auto allocate_temporary_buffer(cuda::std::size_t size) const
  {
    using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<U>;
    using deleter_type   = cuco::detail::custom_deleter<cuda::std::size_t, allocator_type>;

    allocator_type allocator{alloc_};
    auto* const data = size == 0 ? nullptr : allocator.allocate(size, stream_);
    // Destruction enqueues the deallocation on the build stream, after all previously submitted
    // work that uses this buffer.
    return std::unique_ptr<U, deleter_type>{data, deleter_type{size, allocator, stream_}};
  }

  [[nodiscard]] cuda::std::size_t compute_workspace_bytes() const
  {
    if (num_indices_ == 0) { return 0; }

    // CUB only uses these pointers to instantiate the requested algorithm while temporary storage
    // is null; its size-query path returns before launching work or dereferencing them.
    auto* const index_buffer_a   = static_cast<cuda::std::uint32_t*>(nullptr);
    auto* const index_buffer_b   = static_cast<cuda::std::uint32_t*>(nullptr);
    auto* const container_starts = static_cast<cuda::std::int64_t*>(nullptr);
    auto* const payload_offsets  = static_cast<cuda::std::uint32_t*>(nullptr);
    auto* const num_selected     = static_cast<cuda::std::int64_t*>(nullptr);
    auto* const state            = static_cast<roaring_bitmap_build_state*>(nullptr);
    auto const counting_begin    = cuda::counting_iterator<cuda::std::int64_t>{0};

    // After the last CUB operation, the same allocation becomes the array/bitset container queue.
    cuda::std::size_t result         = num_container_slots_ * sizeof(cuda::std::uint32_t);
    cuda::std::size_t required_bytes = 0;

    CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                                required_bytes,
                                                payload_offsets,
                                                payload_offsets,
                                                num_container_slots_,
                                                stream_.get()));
    result = std::max(result, required_bytes);

    required_bytes = 0;
    // Container discovery consumes the caller's iterator only when no normalization is required.
    // Otherwise it consumes one of the materialized uint32_t buffers allocated by build().
    if (input_order_ == roaring_bitmap_builder_input_order::sorted_unique) {
      CUCO_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                          required_bytes,
                                          counting_begin,
                                          container_starts,
                                          num_selected,
                                          num_indices_,
                                          is_container_start{first_, state},
                                          stream_.get()));
    } else {
      CUCO_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                          required_bytes,
                                          counting_begin,
                                          container_starts,
                                          num_selected,
                                          num_indices_,
                                          is_container_start{index_buffer_a, state},
                                          stream_.get()));
    }
    result = std::max(result, required_bytes);

    if (input_order_ != roaring_bitmap_builder_input_order::sorted_unique) {
      required_bytes = 0;
      if (input_order_ == roaring_bitmap_builder_input_order::sorted) {
        CUCO_CUDA_TRY(cub::DeviceSelect::Unique(nullptr,
                                                required_bytes,
                                                first_,
                                                index_buffer_a,
                                                num_selected,
                                                num_indices_,
                                                stream_.get()));
      } else {
        CUCO_CUDA_TRY(cub::DeviceSelect::Unique(nullptr,
                                                required_bytes,
                                                index_buffer_a,
                                                index_buffer_b,
                                                num_selected,
                                                num_indices_,
                                                stream_.get()));
      }
      result = std::max(result, required_bytes);
    }

    if (input_order_ == roaring_bitmap_builder_input_order::unsorted) {
      required_bytes = 0;
      cub::DoubleBuffer<cuda::std::uint32_t> indices{index_buffer_a, index_buffer_b};
      CUCO_CUDA_TRY(cub::DeviceRadixSort::SortKeys(nullptr,
                                                   required_bytes,
                                                   indices,
                                                   num_indices_,
                                                   0,
                                                   sizeof(cuda::std::uint32_t) * 8,
                                                   stream_.get()));
      result = std::max(result, required_bytes);
    }

    return result;
  }

  [[nodiscard]] storage_type build_unsorted()
  {
    auto indices_a =
      allocate_temporary_buffer<cuda::std::uint32_t>(static_cast<cuda::std::size_t>(num_indices_));
    auto indices_b =
      allocate_temporary_buffer<cuda::std::uint32_t>(static_cast<cuda::std::size_t>(num_indices_));
    auto container_starts = allocate_temporary_buffer<cuda::std::int64_t>(num_container_slots_);
    auto state            = allocate_temporary_buffer<roaring_bitmap_build_state>(1);
    auto workspace        = allocate_temporary_buffer<cuda::std::byte>(workspace_bytes_);

    auto const sorted = sort_indices(first_, indices_a.get(), indices_b.get(), workspace.get());
    // Deduplication writes into the inactive radix-sort buffer. Once it completes, the sorted input
    // buffer is no longer needed and is reused for payload offsets.
    deduplicate_indices(sorted.indices, sorted.available_buffer, state.get(), workspace.get());
    return serialize_sorted_unique_indices(sorted.available_buffer,
                                           container_starts.get(),
                                           sorted.indices,
                                           state.get(),
                                           workspace.get());
  }

  [[nodiscard]] storage_type build_sorted()
  {
    auto unique_indices =
      allocate_temporary_buffer<cuda::std::uint32_t>(static_cast<cuda::std::size_t>(num_indices_));
    auto container_starts = allocate_temporary_buffer<cuda::std::int64_t>(num_container_slots_);
    auto payload_offsets  = allocate_temporary_buffer<cuda::std::uint32_t>(num_container_slots_);
    auto state            = allocate_temporary_buffer<roaring_bitmap_build_state>(1);
    auto workspace        = allocate_temporary_buffer<cuda::std::byte>(workspace_bytes_);

    deduplicate_indices(first_, unique_indices.get(), state.get(), workspace.get());
    return serialize_sorted_unique_indices(unique_indices.get(),
                                           container_starts.get(),
                                           payload_offsets.get(),
                                           state.get(),
                                           workspace.get());
  }

  [[nodiscard]] storage_type build_sorted_unique()
  {
    auto container_starts = allocate_temporary_buffer<cuda::std::int64_t>(num_container_slots_);
    auto payload_offsets  = allocate_temporary_buffer<cuda::std::uint32_t>(num_container_slots_);
    auto state            = allocate_temporary_buffer<roaring_bitmap_build_state>(1);
    auto workspace        = allocate_temporary_buffer<cuda::std::byte>(workspace_bytes_);

    CUCO_CUDA_TRY(cudaMemcpyAsync(&state->num_keys,
                                  &num_indices_,
                                  sizeof(num_indices_),
                                  cudaMemcpyHostToDevice,
                                  stream_.get()));
    return serialize_sorted_unique_indices(
      first_, container_starts.get(), payload_offsets.get(), state.get(), workspace.get());
  }

  template <class SourceIt>
  [[nodiscard]] sorted_indices_result sort_indices(SourceIt first,
                                                   cuda::std::uint32_t* indices_a,
                                                   cuda::std::uint32_t* indices_b,
                                                   cuda::std::byte* workspace) const
  {
    CUCO_CUDA_TRY(cub::DeviceTransform::Transform(
      first, indices_a, num_indices_, cuda::std::identity{}, stream_.get()));

    cub::DoubleBuffer<cuda::std::uint32_t> indices{indices_a, indices_b};
    auto workspace_bytes = workspace_bytes_;
    CUCO_CUDA_TRY(cub::DeviceRadixSort::SortKeys(workspace,
                                                 workspace_bytes,
                                                 indices,
                                                 num_indices_,
                                                 0,
                                                 sizeof(cuda::std::uint32_t) * 8,
                                                 stream_.get()));
    return {indices.Current(), indices.Alternate()};
  }

  template <class SourceIt>
  void deduplicate_indices(SourceIt first,
                           cuda::std::uint32_t* unique_indices,
                           roaring_bitmap_build_state* state,
                           cuda::std::byte* workspace) const
  {
    auto workspace_bytes = workspace_bytes_;
    CUCO_CUDA_TRY(cub::DeviceSelect::Unique(workspace,
                                            workspace_bytes,
                                            first,
                                            unique_indices,
                                            &state->num_keys,
                                            num_indices_,
                                            stream_.get()));
  }

  template <class SourceIt>
  [[nodiscard]] storage_type serialize_sorted_unique_indices(SourceIt first,
                                                             cuda::std::int64_t* container_starts,
                                                             cuda::std::uint32_t* payload_offsets,
                                                             roaring_bitmap_build_state* state,
                                                             cuda::std::byte* workspace) const
  {
    analyze_containers(first, container_starts, payload_offsets, state, workspace);
    auto const host_state         = read_build_state(state);
    auto* const container_indexes = reinterpret_cast<cuda::std::uint32_t*>(workspace);
    return write_serialized_bitmap(
      first, host_state, container_starts, payload_offsets, container_indexes);
  }

  template <class SourceIt>
  void analyze_containers(SourceIt first,
                          cuda::std::int64_t* container_starts,
                          cuda::std::uint32_t* payload_offsets,
                          roaring_bitmap_build_state* state,
                          cuda::std::byte* workspace) const
  {
    auto const counting_begin = cuda::counting_iterator<cuda::std::int64_t>{0};
    auto workspace_bytes      = workspace_bytes_;
    CUCO_CUDA_TRY(cub::DeviceSelect::If(workspace,
                                        workspace_bytes,
                                        counting_begin,
                                        container_starts,
                                        &state->num_containers,
                                        num_indices_,
                                        is_container_start{first, state},
                                        stream_.get()));

    compute_container_payload_sizes<<<cuco::detail::grid_size(num_container_slots_),
                                      cuco::detail::default_block_size(),
                                      0,
                                      stream_.get()>>>(
      payload_offsets, num_container_slots_, container_starts, state);
    workspace_bytes = workspace_bytes_;
    CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(workspace,
                                                workspace_bytes,
                                                payload_offsets,
                                                payload_offsets,
                                                num_container_slots_,
                                                stream_.get()));

    // No later CUB operation needs temporary storage, so reuse it as the container work queue.
    auto* const container_indexes = reinterpret_cast<cuda::std::uint32_t*>(workspace);
    CUCO_CUDA_TRY(
      cudaMemsetAsync(&state->num_array_containers,
                      0,
                      sizeof(state->num_array_containers) + sizeof(state->num_bitset_containers),
                      stream_.get()));
    collect_container_indexes<<<cuco::detail::grid_size(num_container_slots_),
                                cuco::detail::default_block_size(),
                                0,
                                stream_.get()>>>(
      container_indexes, num_container_slots_, container_starts, state);

    compute_roaring_bitmap_build_size<<<1, 1, 0, stream_.get()>>>(
      state, container_starts, payload_offsets);
    CUCO_CUDA_TRY(cudaPeekAtLastError());
  }

  [[nodiscard]] roaring_bitmap_build_state read_build_state(
    roaring_bitmap_build_state const* state) const
  {
    using metadata_type = roaring_bitmap_metadata<cuda::std::uint32_t>;

    roaring_bitmap_build_state host_state{};
    CUCO_CUDA_TRY(cuco::detail::memcpy_async(
      &host_state, state, sizeof(host_state), cudaMemcpyDeviceToHost, stream_));
    // The serialized allocation size depends on device-computed container cardinalities. This is
    // the only synchronization required before final storage can be allocated.
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    stream_.sync();
#else
    stream_.wait();
#endif

    CUCO_EXPECTS(host_state.num_containers >= 0 &&
                   host_state.num_containers <= metadata_type::max_num_containers,
                 "Invalid generated container count");
    CUCO_EXPECTS(host_state.num_keys >= 0, "Invalid generated index count");
    CUCO_EXPECTS(host_state.num_array_containers + host_state.num_bitset_containers ==
                   static_cast<cuda::std::uint64_t>(host_state.num_containers),
                 "Invalid generated container indexes");
    return host_state;
  }

  template <class SourceIt>
  [[nodiscard]] storage_type write_serialized_bitmap(SourceIt first,
                                                     roaring_bitmap_build_state const& host_state,
                                                     cuda::std::int64_t* container_starts,
                                                     cuda::std::uint32_t* payload_offsets,
                                                     cuda::std::uint32_t* container_indexes) const
  {
    using metadata_type = typename storage_type::metadata_type;

    auto storage = storage_type{
      metadata_type::from_no_run_build(host_state.size_bytes,
                                       static_cast<cuda::std::size_t>(host_state.num_keys),
                                       static_cast<cuda::std::int32_t>(host_state.num_containers)),
      alloc_,
      stream_};

    auto const header_items = std::max<cuda::std::int64_t>(1, host_state.num_containers);
    write_roaring_bitmap_header<<<cuco::detail::grid_size(header_items),
                                  cuco::detail::default_block_size(),
                                  0,
                                  stream_.get()>>>(
      storage.data(), first, host_state, container_starts, payload_offsets);

    if (host_state.num_containers > 0) {
      constexpr cuda::std::uint32_t block_size      = 256;
      constexpr cuda::std::uint32_t warps_per_block = block_size / 32;
      constexpr cuda::std::uint32_t bitset_blocks_per_container =
        metadata_type::bitset_container_bytes / sizeof(cuda::std::uint64_t) / block_size;
      auto const array_blocks =
        (host_state.num_array_containers + warps_per_block - 1) / warps_per_block;
      auto const bitset_blocks = host_state.num_bitset_containers * bitset_blocks_per_container;
      // Array indexes grow from the front of the queue and bitset indexes from the back. Their
      // internal order is irrelevant because each entry names its destination container.
      auto* const bitset_containers =
        container_indexes + num_container_slots_ - host_state.num_bitset_containers;

      write_roaring_containers<block_size>
        <<<array_blocks + bitset_blocks, block_size, 0, stream_.get()>>>(storage.data(),
                                                                         first,
                                                                         host_state,
                                                                         container_starts,
                                                                         payload_offsets,
                                                                         container_indexes,
                                                                         bitset_containers);
    }
    CUCO_CUDA_TRY(cudaPeekAtLastError());
    return storage;
  }

  [[nodiscard]] static constexpr roaring_bitmap_build_state empty_build_state() noexcept
  {
    return {0, 0, 2 * sizeof(cuda::std::uint32_t), 0, 0};
  }

  InputIt first_;
  cuda::std::int64_t num_indices_;
  cuda::std::size_t num_container_slots_;
  roaring_bitmap_builder_input_order input_order_;
  Allocator alloc_;
  cuda::stream_ref stream_;
  cuda::std::size_t workspace_bytes_;
};

}  // namespace cuco::experimental::detail
