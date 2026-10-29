/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/utility/allocator.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <unordered_map>

namespace cuco::benchmark {

/**
 * @brief Tracks peak outstanding device bytes allocated through tracking_allocator in this scope.
 *
 * Construct after benchmark setup, immediately before state.exec or a separate measurement pass.
 * Allocations made before the scope, including input/output buffers and existing container storage,
 * are excluded. Accounting follows allocator calls, not CUDA pool reservations or physical
 * completion of asynchronous frees. Each benchmark state runs on one host thread. Nested logging
 * scopes are not supported.
 */
class memory_stats_logger {
 public:
  memory_stats_logger()
  {
    CUCO_EXPECTS(active_ == nullptr, "Nested memory statistics scopes are not supported");
    active_ = this;
  }

  ~memory_stats_logger() { active_ = nullptr; }

  memory_stats_logger(memory_stats_logger const&)            = delete;
  memory_stats_logger& operator=(memory_stats_logger const&) = delete;
  memory_stats_logger(memory_stats_logger&&)                 = delete;
  memory_stats_logger& operator=(memory_stats_logger&&)      = delete;

  [[nodiscard]] std::size_t peak_memory_usage() const noexcept { return peak_bytes_; }

 private:
  template <class, class>
  friend class tracking_allocator;

  void allocate(void* pointer, std::size_t bytes)
  {
    allocations_.emplace(pointer, bytes);
    current_bytes_ += bytes;
    peak_bytes_ = std::max(peak_bytes_, current_bytes_);
  }

  void deallocate(void* pointer)
  {
    auto const allocation = allocations_.find(pointer);
    // Rehash can free storage allocated during setup, before this scope started.
    if (allocation != allocations_.end()) {
      current_bytes_ -= allocation->second;
      allocations_.erase(allocation);
    }
  }

  inline static thread_local memory_stats_logger* active_ = nullptr;
  std::unordered_map<void*, std::size_t> allocations_;
  std::size_t current_bytes_ = 0;
  std::size_t peak_bytes_    = 0;
};

/**
 * @brief Allocator adaptor preserving cuco's stream-ordered allocation behavior.
 *
 * Copies and rebound allocators consult the active logger at allocation time, so containers
 * constructed before the logger still report temporary storage allocated by their operations.
 */
template <class T, class Upstream = cuco::cuda_allocator<T>>
class tracking_allocator {
 public:
  using value_type = T;

  template <class U>
  struct rebind {
    using other =
      tracking_allocator<U, typename std::allocator_traits<Upstream>::template rebind_alloc<U>>;
  };

  tracking_allocator() = default;

  template <class U, class OtherUpstream>
  tracking_allocator(tracking_allocator<U, OtherUpstream> const& other) : upstream_{other.upstream_}
  {
  }

  T* allocate(std::size_t count, cuda::stream_ref stream)
  {
    auto* pointer = upstream_.allocate(count, stream);
    if (auto* logger = memory_stats_logger::active_; logger != nullptr && count != 0) {
      try {
        logger->allocate(pointer, count * sizeof(T));
      } catch (...) {
        upstream_.deallocate(pointer, count, stream);
        throw;
      }
    }
    return pointer;
  }

  void deallocate(T* pointer, std::size_t count, cuda::stream_ref stream)
  {
    upstream_.deallocate(pointer, count, stream);
    if (auto* logger = memory_stats_logger::active_; logger != nullptr) {
      logger->deallocate(pointer);
    }
  }

  template <class U, class OtherUpstream>
  bool operator==(tracking_allocator<U, OtherUpstream> const& other) const noexcept
  {
    return upstream_ == other.upstream_;
  }

  template <class U, class OtherUpstream>
  bool operator!=(tracking_allocator<U, OtherUpstream> const& other) const noexcept
  {
    return not(*this == other);
  }

 private:
  template <class, class>
  friend class tracking_allocator;

  Upstream upstream_;
};

}  // namespace cuco::benchmark
