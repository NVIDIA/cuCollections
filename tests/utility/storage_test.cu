/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cuco/aow_storage.cuh>
#include <cuco/extent.cuh>
#include <cuco/pair.cuh>
#include <cuco/utility/allocator.hpp>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("Storage tests",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int64_t))
{
  constexpr std::size_t size{1'000};
  constexpr int window_size{2};
  constexpr std::size_t gold_capacity{2'000};

  using allocator_type = cuco::cuda_allocator<char>;
  auto allocator       = allocator_type{};

  SECTION("Initialize empty storage is allowed.")
  {
    auto s = cuco::
      aow_storage<cuco::pair<Key, Value>, window_size, cuco::extent<std::size_t>, allocator_type>{
        cuco::extent<std::size_t>{0}, allocator};

    s.initialize(cuco::pair<Key, Value>{1, 1});
  }

  SECTION("Allocate array of pairs with AoS storage.")
  {
    auto s = cuco::
      aow_storage<cuco::pair<Key, Value>, window_size, cuco::extent<std::size_t>, allocator_type>(
        cuco::extent{size}, allocator);
    auto const num_windows = s.num_windows();
    auto const capacity    = s.capacity();

    REQUIRE(num_windows == size);
    REQUIRE(capacity == gold_capacity);
  }

  SECTION("Allocate array of pairs with AoS storage with static extent.")
  {
    using extent_type = cuco::extent<std::size_t, size>;
    auto s = cuco::aow_storage<cuco::pair<Key, Value>, window_size, extent_type, allocator_type>(
      extent_type{}, allocator);
    auto const num_windows = s.num_windows();
    auto const capacity    = s.capacity();

    STATIC_REQUIRE(num_windows == size);
    STATIC_REQUIRE(capacity == gold_capacity);
  }

  SECTION("Allocate array of keys with AoS storage.")
  {
    auto s = cuco::aow_storage<Key, window_size, cuco::extent<std::size_t>, allocator_type>(
      cuco::extent{size}, allocator);
    auto const num_windows = s.num_windows();
    auto const capacity    = s.capacity();

    REQUIRE(num_windows == size);
    REQUIRE(capacity == gold_capacity);
  }

  SECTION("Allocate array of keys with AoS storage with static extent.")
  {
    using extent_type = cuco::extent<std::size_t, size>;
    auto s =
      cuco::aow_storage<Key, window_size, extent_type, allocator_type>(extent_type{}, allocator);
    auto const num_windows = s.num_windows();
    auto const capacity    = s.capacity();

    STATIC_REQUIRE(num_windows == size);
    STATIC_REQUIRE(capacity == gold_capacity);
  }
}
