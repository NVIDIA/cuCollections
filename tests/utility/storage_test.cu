/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

#include <test_utils.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>
#include <cuco/pair.cuh>
#include <cuco/utility/allocator.hpp>

#include <cuda/std/bit>

#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

TEMPLATE_TEST_CASE_SIG("utility storage tests",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int64_t))
{
  constexpr std::size_t size{1'000};
  constexpr int bucket_size{2};
  constexpr std::size_t gold_capacity{1'000};

  using allocator_type = cuco::cuda_allocator<char>;
  auto allocator       = allocator_type{};

  SECTION("Initialize empty storage is allowed.")
  {
    auto s = cuco::bucket_storage<cuco::pair<Key, Value>,
                                  bucket_size,
                                  cuco::extent<std::size_t>,
                                  allocator_type>{
      cuco::extent<std::size_t>{0}, allocator, cuda::stream_ref{cudaStream_t{nullptr}}};

    s.initialize(cuco::pair<Key, Value>{1, 1});
  }

  SECTION("Allocate array of pairs with AoS storage.")
  {
    auto s = cuco::bucket_storage<cuco::pair<Key, Value>,
                                  bucket_size,
                                  cuco::extent<std::size_t>,
                                  allocator_type>(
      cuco::extent{size}, allocator, cuda::stream_ref{cudaStream_t{nullptr}});
    auto const num_buckets = s.num_buckets();
    auto const capacity    = s.capacity();

    REQUIRE(num_buckets == size / bucket_size);
    REQUIRE(capacity == gold_capacity);
  }

  SECTION("Allocate array of pairs with AoS storage with static extent.")
  {
    using extent_type = cuco::extent<std::size_t, size>;
    auto s = cuco::bucket_storage<cuco::pair<Key, Value>, bucket_size, extent_type, allocator_type>(
      extent_type{}, allocator, cuda::stream_ref{cudaStream_t{nullptr}});
    auto const num_buckets = s.num_buckets();
    auto const capacity    = s.capacity();

    STATIC_REQUIRE(num_buckets == size / bucket_size);
    STATIC_REQUIRE(capacity == gold_capacity);
  }

  SECTION("Allocate array of keys with AoS storage.")
  {
    auto s = cuco::bucket_storage<Key, bucket_size, cuco::extent<std::size_t>, allocator_type>(
      cuco::extent{size}, allocator, cuda::stream_ref{cudaStream_t{nullptr}});
    auto const num_buckets = s.num_buckets();
    auto const capacity    = s.capacity();

    REQUIRE(num_buckets == size / bucket_size);
    REQUIRE(capacity == gold_capacity);
  }

  SECTION("Allocate array of keys with AoS storage with static extent.")
  {
    using extent_type = cuco::extent<std::size_t, size>;
    auto s            = cuco::bucket_storage<Key, bucket_size, extent_type, allocator_type>(
      extent_type{}, allocator, cuda::stream_ref{cudaStream_t{nullptr}});
    auto const num_buckets = s.num_buckets();
    auto const capacity    = s.capacity();

    STATIC_REQUIRE(num_buckets == size / bucket_size);
    STATIC_REQUIRE(capacity == gold_capacity);
  }

  SECTION("Storage alignment constant is correct for pairs.")
  {
    using storage_ref_type =
      cuco::bucket_storage_ref<cuco::pair<Key, Value>, bucket_size, cuco::extent<std::size_t>>;
    using bucket_type = typename storage_ref_type::bucket_type;

    constexpr auto alignment      = storage_ref_type::alignment;
    constexpr auto expected_align = cuda::std::min(cuda::std::bit_ceil(sizeof(bucket_type)),
                                                   storage_ref_type::max_vector_load_bytes);

    STATIC_REQUIRE(alignment == expected_align);
    STATIC_REQUIRE(cuda::std::has_single_bit(alignment));
  }

  SECTION("Storage alignment constant is correct for keys.")
  {
    using storage_ref_type = cuco::bucket_storage_ref<Key, bucket_size, cuco::extent<std::size_t>>;
    using bucket_type      = typename storage_ref_type::bucket_type;

    constexpr auto alignment      = storage_ref_type::alignment;
    constexpr auto expected_align = cuda::std::min(cuda::std::bit_ceil(sizeof(bucket_type)),
                                                   storage_ref_type::max_vector_load_bytes);

    STATIC_REQUIRE(alignment == expected_align);
    STATIC_REQUIRE(cuda::std::has_single_bit(alignment));
  }

  SECTION("Storage data pointer is aligned to bucket boundary for pairs.")
  {
    auto s = cuco::bucket_storage<cuco::pair<Key, Value>,
                                  bucket_size,
                                  cuco::extent<std::size_t>,
                                  allocator_type>(
      cuco::extent{size}, allocator, cuda::stream_ref{cudaStream_t{nullptr}});

    auto const ptr       = reinterpret_cast<std::uintptr_t>(s.data());
    auto const alignment = decltype(s)::ref_type::alignment;

    REQUIRE((ptr % alignment) == 0);
  }

  SECTION("Storage data pointer is aligned to bucket boundary for keys.")
  {
    auto s = cuco::bucket_storage<Key, bucket_size, cuco::extent<std::size_t>, allocator_type>(
      cuco::extent{size}, allocator, cuda::stream_ref{cudaStream_t{nullptr}});

    auto const ptr       = reinterpret_cast<std::uintptr_t>(s.data());
    auto const alignment = decltype(s)::ref_type::alignment;

    REQUIRE((ptr % alignment) == 0);
  }
}

TEMPLATE_TEST_CASE_SIG("bucket storage alignment with different bucket sizes",
                       "",
                       ((typename T, int BucketSize), T, BucketSize),
                       (int32_t, 1),
                       (int32_t, 2),
                       (int32_t, 4),
                       (int64_t, 1),
                       (int64_t, 2),
                       (cuco::pair<int32_t, int32_t>, 1),
                       (cuco::pair<int32_t, int32_t>, 2),
                       (cuco::pair<int64_t, int64_t>, 1))
{
  constexpr std::size_t size{1'000};

  using allocator_type = cuco::cuda_allocator<char>;
  using storage_type =
    cuco::bucket_storage<T, BucketSize, cuco::extent<std::size_t>, allocator_type>;
  using storage_ref_type = typename storage_type::ref_type;

  auto allocator = allocator_type{};

  SECTION("Data pointer is aligned to bucket boundary.")
  {
    auto s = storage_type(cuco::extent{size}, allocator, cuda::stream_ref{cudaStream_t{nullptr}});

    auto const ptr       = reinterpret_cast<std::uintptr_t>(s.data());
    auto const alignment = storage_ref_type::alignment;

    REQUIRE((ptr % alignment) == 0);
  }
}
